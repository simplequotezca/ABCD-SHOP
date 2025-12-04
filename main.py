import os
import json
import base64
import uuid
import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()

# ------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------
# LOAD SHOPS
# ------------------------------------------------------
def load_shops():
    try:
        return {shop["webhook_token"]: shop for shop in json.loads(SHOPS_JSON)}
    except:
        return {}

SHOPS = load_shops()

# ------------------------------------------------------
# DOWNLOAD TWILIO IMAGE
# ------------------------------------------------------
async def download_twilio_image(url: str) -> bytes:
    TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

# ------------------------------------------------------
# AI DAMAGE ESTIMATION
# ------------------------------------------------------
async def analyze_damage(images_b64: list, shop_name: str):
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
You are an elite automotive damage estimator. 
Return extremely accurate severity, areas, and pricing.
Always use driver's POV for left/right.
Return JSON ONLY:
{{
 "severity": "...",
 "estimated_cost_min": ...,
 "estimated_cost_max": ...,
 "areas": [...],
 "types": [...],
 "narrative": "..."
}}
                """,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze damage for {shop_name}."}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in images_b64
                ],
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return data

    except Exception as e:
        return {
            "severity": "unknown",
            "estimated_cost_min": 0,
            "estimated_cost_max": 0,
            "areas": [],
            "types": [],
            "narrative": f"AI error: {str(e)}",
        }

# ------------------------------------------------------
# HUMAN-FRIENDLY ESTIMATE TEXT
# ------------------------------------------------------
def build_estimate_message(shop_name: str, est: dict):
    return f"""
🛠 AI Damage Estimate for {shop_name}

Severity: {est['severity']}
Estimated Cost: ${est['estimated_cost_min']:,} – ${est['estimated_cost_max']:,}
Areas: {", ".join(est["areas"])}
Damage Types: {", ".join(est["types"])}

{est["narrative"]}

Reply 1 to confirm this looks accurate, or 2 to send more photos.

To book an appointment, reply:
BOOK + your name, phone #, email,
make & model of your car, and
preferred date/time (any order).
""".strip()

# ------------------------------------------------------
# BOOKING PARSER
# ------------------------------------------------------
def parse_booking(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    name = phone = email = car = appt = ""

    for l in lines:
        if "@" in l:
            email = l
        elif any(x.isdigit() for x in l) and "-" in l:
            phone = l
        elif any(x in l.lower() for x in ["am", "pm", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "dec", "jan", "feb"]):
            appt = l
        elif any(x in l.lower() for x in ["toyota", "honda", "bmw", "benz", "nissan"]):
            car = l
        else:
            name = l

    return name, phone, email, car, appt

# ------------------------------------------------------
# CREATE CALENDAR EVENT
# ------------------------------------------------------
def create_calendar_event(shop: dict, name, phone, email, car, appt, est, image_links):
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")),
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": f"Appointment – {shop['name']} – {name}",
        "description": f"""
Name: {name}
Phone: {phone}
Email: {email}
Vehicle: {car}

Requested: {appt}

AI Estimate:
Severity: {est['severity']} | Estimate: ${est['estimated_cost_min']:,} – ${est['estimated_cost_max']:,}
Areas: {", ".join(est["areas"])}
Types: {", ".join(est["types"])}

Customer Photos:
{image_links}
""",
        "start": {"dateTime": convert_to_utc(appt), "timeZone": "America/Toronto"},
        "end": {"dateTime": convert_to_utc(appt, plus_minutes=30), "timeZone": "America/Toronto"},
    }

    result = service.events().insert(calendarId=shop["calendar_id"], body=event).execute()
    return result.get("id")

# ------------------------------------------------------
# TIMEZONE FIX
# ------------------------------------------------------
import pytz
from dateutil import parser as dateparser

def convert_to_utc(text, plus_minutes=0):
    local = pytz.timezone("America/Toronto")
    dt = dateparser.parse(text)
    dt = dt.replace(tzinfo=local)
    dt = dt.astimezone(pytz.utc)
    dt = dt + timedelta(minutes=plus_minutes)
    return dt.isoformat()

# ------------------------------------------------------
# WEBHOOK
# ------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    incoming_text = (form.get("Body") or "").strip()
    token = request.query_params.get("token")

    resp = MessagingResponse()

    if token not in SHOPS:
        resp.message("Invalid shop token.")
        return Response(content=str(resp), media_type="application/xml")

    shop = SHOPS[token]

    # --------------------------------------------------
    # 1) User sends images → generate estimate (single message!)
    # --------------------------------------------------
    num_media = int(form.get("NumMedia", "0"))
    if num_media > 0:
        images_b64 = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            content = await download_twilio_image(url)
            images_b64.append(base64.b64encode(content).decode())

        est = await analyze_damage(images_b64, shop["name"])
        estimate_msg = build_estimate_message(shop["name"], est)

        # Store state per user (simple memory)
        USER_STATE[form.get("From")] = {"stage": "confirm_estimate", "estimate": est, "images_b64": images_b64}

        resp.message(estimate_msg)
        return Response(content=str(resp), media_type="application/xml")

    # --------------------------------------------------
    # 2) FIXED → "1" ALWAYS COUNTS AS CONFIRMATION
    # --------------------------------------------------
    state = USER_STATE.get(form.get("From"))

    if state and state.get("stage") == "confirm_estimate":
        if incoming_text.startswith("1"):  # <-- FIXED
            resp.message("✅ Thanks! The shop will review your estimate shortly.\n\nTo book now, reply: BOOK + info.")
            USER_STATE[form.get("From")]["stage"] = "await_booking"
            return Response(content=str(resp), media_type="application/xml")

        if incoming_text.startswith("2"):
            resp.message("No problem — please send 1–3 new photos.")
            USER_STATE[form.get("From")] = {"stage": "await_photos"}
            return Response(content=str(resp), media_type="application/xml")

    # --------------------------------------------------
    # 3) BOOKING
    # --------------------------------------------------
    if incoming_text.lower().startswith("book"):
        name, phone, email, car, appt = parse_booking(incoming_text)
        est = state.get("estimate")
        images = state.get("images_b64", [])

        # Build hosted image links
        image_links = ""
        for img in images:
            fname = f"/img/{uuid.uuid4()}.jpg"
            IMAGE_STORE[fname] = img
            full_url = f"{os.getenv('APP_URL')}{fname}"
            image_links += f"{full_url}\n"

        event_id = create_calendar_event(shop, name, phone, email, car, appt, est, image_links)
        resp.message(f"✅ Thanks {name}! Your request was sent.\n📅 Requested: {appt}")
        return Response(content=str(resp), media_type="application/xml")

    # --------------------------------------------------
    # Default
    # --------------------------------------------------
    resp.message("📸 Welcome! Please send photos to begin.")
    return Response(content=str(resp), media_type="application/xml")

# ------------------------------------------------------
# SIMPLE IMAGE HOSTING ENDPOINT
# ------------------------------------------------------
IMAGE_STORE = {}
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import timedelta

@app.get("/img/{img_id}")
async def serve_img(img_id: str):
    b64 = IMAGE_STORE.get(f"/img/{img_id}")
    if not b64:
        return Response(status_code=404)
    raw = base64.b64decode(b64)
    return StreamingResponse(BytesIO(raw), media_type="image/jpeg")

# ------------------------------------------------------
# USER STATE
# ------------------------------------------------------
USER_STATE = {}
