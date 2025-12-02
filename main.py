import os
import json
import re
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from dateutil import parser as date_parser
import pytz

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_TZ = "America/Toronto"
TZ = pytz.timezone(DEFAULT_TZ)

# ============================================================
# SHOP LOADING
# ============================================================

class ShopConfig:
    def __init__(self, id, name, webhook_token, calendar_id=None):
        self.id = id
        self.name = name
        self.webhook_token = webhook_token
        self.calendar_id = calendar_id

def load_shops():
    out = {}
    raw = os.getenv("SHOPS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            for s in data:
                out[s["webhook_token"]] = ShopConfig(
                    id=s.get("id"),
                    name=s.get("name"),
                    webhook_token=s["webhook_token"],
                    calendar_id=s.get("calendar_id")
                )
        except Exception as e:
            print("SHOPS_JSON parse error:", e)
    return out

SHOPS_BY_TOKEN = load_shops()

# ============================================================
# GOOGLE CALENDAR INIT
# ============================================================

CAL_SCOPES = ["https://www.googleapis.com/auth/calendar"]
calendar_service = None

def init_calendar():
    global calendar_service
    creds_raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_raw:
        print("No Google credentials.")
        return
    try:
        info = json.loads(creds_raw)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=CAL_SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        print("Calendar ready.")
    except Exception as e:
        print("Calendar init error:", e)

init_calendar()

# ============================================================
# ANONYMOUS IMAGE UPLOAD (0x0.st)
# ============================================================

async def upload_image_anonymous(image_bytes: bytes) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as client:
            files = {'file': ('upload.jpg', image_bytes, 'image/jpeg')}
            r = await client.post("https://0x0.st", files=files)
            if r.status_code == 200:
                return r.text.strip()
            print("Upload failed:", r.text)
    except Exception as e:
        print("Upload error:", e)
    return None

# ============================================================
# TWILIO IMAGE DOWNLOAD
# ============================================================

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

# ============================================================
# DAMAGE ANALYSIS (UNCHANGED)
# ============================================================

async def analyze_damage(image_bytes_list: List[bytes]):
    system_prompt = (
        "Return STRICT JSON only:\n"
        "{ \"severity\": \"minor|moderate|severe\", "
        "\"estimated_cost_min\": num, \"estimated_cost_max\": num, "
        "\"damaged_areas\": [], \"damage_types\": [], "
        "\"summary\": \"2-4 sentences\" }"
    )

    content_list = [
        {"type": "text", "text": "Analyze the vehicle damage in the attached images."}
    ]

    for img in image_bytes_list:
        b64 = base64.b64encode(img).decode()
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list}
            ],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("AI ERROR:", e)
        return None

# ============================================================
# BOOKING PARSER
# ============================================================

VIN_REGEX = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

def extract_first(regex, text):
    m = regex.search(text)
    if not m:
        return None, text
    return m.group(), text[:m.start()] + " " + text[m.end():]

def parse_booking_message(body: str):
    original = body.strip()
    cleaned = re.sub(r"\bbook\b", "", original, flags=re.IGNORECASE).strip()

    vin, cleaned = extract_first(VIN_REGEX, cleaned)
    email, cleaned = extract_first(EMAIL_REGEX, cleaned)
    phone, cleaned = extract_first(PHONE_REGEX, cleaned)

    dt = None
    try:
        parsed = date_parser.parse(cleaned, fuzzy=True)
        dt = TZ.localize(parsed) if parsed.tzinfo is None else parsed.astimezone(TZ)
    except:
        dt = None

    parts = cleaned.split()
    name = " ".join(parts[:2]) if len(parts) >= 2 else None
    vehicle = " ".join(parts[2:]) if len(parts) >= 3 else None

    return {
        "raw": original,
        "name": name,
        "email": email,
        "phone": phone,
        "vehicle": vehicle,
        "vin": vin,
        "preferred_dt": dt
    }
  # ============================================================
# CALENDAR EVENT CREATION
# ============================================================

def create_calendar_event(shop: ShopConfig, title, description,
                          start_dt: Optional[datetime] = None):
    if not calendar_service:
        print("No calendar service.")
        return
    if not shop.calendar_id:
        print("No calendar ID for shop.")
        return

    if start_dt is None:
        start_dt = TZ.localize(datetime.now())

    end_dt = start_dt + timedelta(minutes=30)

    event = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": DEFAULT_TZ
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": DEFAULT_TZ
        }
    }

    try:
        created = calendar_service.events().insert(
            calendarId=shop.calendar_id,
            body=event
        ).execute()
        print("Calendar event created:", created.get("id"))
    except Exception as e:
        print("CALENDAR ERROR:", e)

# ============================================================
# ESTIMATE LEAD LOGGING
# ============================================================

def log_estimate_lead(shop: ShopConfig, phone, result: Dict[str, Any]):
    desc = (
        f"Lead (Estimate Only)\n"
        f"Phone: {phone}\n"
        f"Severity: {result['severity']}\n"
        f"Estimate: ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n"
        f"Areas: {', '.join(result['damaged_areas'])}\n"
        f"Types: {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n"
        "NOT BOOKED\n"
    )

    # embed uploaded image URLs
    imgs = result.get("image_urls") or []
    if imgs:
        desc += "\nCustomer Photos:\n"
        for u in imgs:
            desc += f'<img src="{u}" style="width:100%;max-width:450px;"><br>'

    create_calendar_event(shop, "Estimate Lead", desc)

# ============================================================
# BOOKING EVENT LOGGING
# ============================================================

def log_booking_event(shop: ShopConfig, phone, booking, last_estimate):
    name = booking.get("name") or "Unknown"
    vehicle = booking.get("vehicle") or "Unknown"
    vin = booking.get("vin") or "N/A"
    email = booking.get("email") or "N/A"

    lines = [
        f"Name: {name}",
        f"Phone: {phone}",
        f"Email: {email}",
        f"Vehicle: {vehicle}",
        f"VIN: {vin}",
        "",
        f"Original SMS: {booking['raw']}",
        ""
    ]

    imgs = []
    if last_estimate:
        lines.append("AI Estimate:")
        lines.append(
            f"Severity: {last_estimate['severity']} | "
            f"${last_estimate['estimated_cost_min']} – ${last_estimate['estimated_cost_max']}"
        )
        lines.append(f"Areas: {', '.join(last_estimate['damaged_areas'])}")
        lines.append(f"Types: {', '.join(last_estimate['damage_types'])}")
        lines.append("")
        lines.append(last_estimate["summary"])
        imgs = last_estimate.get("image_urls") or []

    if imgs:
        lines.append("")
        lines.append("Customer Photos:")
        for u in imgs:
            lines.append(f'<img src="{u}" style="width:100%;max-width:450px;">')

    description = "\n".join(lines)

    start_dt = booking.get("preferred_dt")
    create_calendar_event(shop, "Booked Appointment", description, start_dt)
  # ============================================================
# MEMORY OF LAST ESTIMATE
# ============================================================

LAST_ESTIMATE_BY_PHONE: Dict[str, Dict[str, Any]] = {}

# ============================================================
# SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        return Response("Invalid shop token", status_code=403)

    form = await request.form()
    num_media = int(form.get("NumMedia", "0"))
    body = (form.get("Body") or "").strip()
    phone = form.get("From") or "Unknown"

    reply = MessagingResponse()

    # -------------------------
    # BOOKING FLOW
    # -------------------------
    if num_media == 0 and re.search(r"\bbook\b", body, flags=re.IGNORECASE):
        booking = parse_booking_message(body)
        last = LAST_ESTIMATE_BY_PHONE.get(phone)
        log_booking_event(shop, phone, booking, last)

        msg = f"✅ Thanks! Your request has been sent to {shop.name}."
        if booking.get("preferred_dt"):
            d = booking["preferred_dt"].strftime("%A, %b %d at %I:%M %p")
            msg += f"\n📅 Requested: {d}"
        reply.message(msg)
        return Response(str(reply), media_type="application/xml")

    # -------------------------
    # NO MEDIA → WELCOME
    # -------------------------
    if num_media == 0:
        reply.message(
            f"📸 Welcome to {shop.name}!\n"
            "Send 1–3 photos of the vehicle damage to receive an AI estimate."
        )
        return Response(str(reply), media_type="application/xml")

    # -------------------------
    # MEDIA RECEIVED → AI ANALYSIS
    # -------------------------
    images = []
    uploaded_urls = []

    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if not url:
            continue

        try:
            img_bytes = await download_twilio_image(url)
            images.append(img_bytes)

            # Upload to 0x0.st → get permanent URL
            public_url = await upload_image_anonymous(img_bytes)
            if public_url:
                uploaded_urls.append(public_url)

        except Exception as e:
            print("Image download/upload error:", e)

    reply.message("📸 Photos received! Analyzing damage…")

    result = await analyze_damage(images)
    if not result:
        reply.message("⚠️ Could not analyze image. Please resend.")
        return Response(str(reply), media_type="application/xml")

    # attach uploaded image URLs
    result["image_urls"] = uploaded_urls

    # store estimate per phone
    LAST_ESTIMATE_BY_PHONE[phone] = result

    # log lead → Calendar
    log_estimate_lead(shop, phone, result)

    # send SMS estimate
    txt = (
        f"🛠 AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {result['severity']}\n"
        f"Estimated Cost: ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n"
        f"Areas: {', '.join(result['damaged_areas'])}\n"
        f"Damage Types: {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n\n"
        "To book an appointment, reply: BOOK + your info."
    )

    reply.message(txt)
    return Response(str(reply), media_type="application/xml")
