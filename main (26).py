import os
import json
import re
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from dateutil import parser as date_parser

from google.oauth2 import service_account
from googleapiclient.discovery import build

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "America/Toronto")

if ZoneInfo is not None:
    try:
        LOCAL_TZ = ZoneInfo(DEFAULT_TZ)
    except Exception:
        print(f"Invalid DEFAULT_TIMEZONE '{DEFAULT_TZ}', falling back to UTC")
        LOCAL_TZ = timezone.utc
else:
    LOCAL_TZ = timezone.utc

class ShopConfig:
    def __init__(self, id: str, name: str, webhook_token: str,
                 calendar_id: Optional[str] = None):
        self.id = id
        self.name = name
        self.webhook_token = webhook_token
        self.calendar_id = calendar_id

def load_shops() -> Dict[str, ShopConfig]:
    shops_by_token: Dict[str, ShopConfig] = {}
    raw = os.getenv("SHOPS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            for s in data:
                shops_by_token[s["webhook_token"]] = ShopConfig(
                    id=s.get("id", s["name"]),
                    name=s.get("name", "Collision Centre"),
                    webhook_token=s["webhook_token"],
                    calendar_id=s.get("calendar_id")
                )
        except Exception as e:
            print("Error parsing SHOPS_JSON:", e)

    if not shops_by_token:
        print("No SHOPS_JSON found – using default Mississauga shop.")
        shops_by_token["shop_miss_123"] = ShopConfig(
            id="miss",
            name="Mississauga Collision Centre",
            webhook_token="shop_miss_123",
            calendar_id=os.getenv("DEFAULT_CALENDAR_ID")
        )
    return shops_by_token

SHOPS_BY_TOKEN = load_shops()

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
calendar_service = None

def init_calendar():
    global calendar_service
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("No Google credentials found.")
        return
    try:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=CALENDAR_SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        print("Google Calendar Ready.")
    except Exception as e:
        print("Calendar init error:", e)

init_calendar()

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

async def analyze_damage(image_bytes_list: List[bytes]) -> Optional[Dict[str, Any]]:
    system_prompt = (
        "You are a professional auto-body damage estimator in Ontario, Canada (2025 prices).\n"
        "You ONLY return STRICT JSON, no extra text.\n\n"
        "ALWAYS follow these rules:\n"
        "- Use the DRIVER'S POINT OF VIEW for left/right.\n"
        "- Carefully inspect all visible areas.\n"
        "- Severe if any structural/suspension.\n"
        "- Summary 2–4 sentences.\n"
    )
    content_list = [{"type": "text", "text": "Analyze the vehicle damage in the attached images."}]
    for img_bytes in image_bytes_list:
        b64 = base64.b64encode(img_bytes).decode()
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{b64}"}
        })
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ],
            max_tokens=600,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("AI ERROR:", e)
        return None

VIN_REGEX = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

def extract_first(regex: re.Pattern, text: str) -> (Optional[str], str):
    m = regex.search(text)
    if not m:
        return None, text
    value = m.group(1 if m.lastindex else 0).strip()
    new_text = text[:m.start()] + " " + text[m.end():]
    return value, new_text

def parse_booking_message(body: str) -> Dict[str, Any]:
    original = body.strip()
    cleaned = re.sub(r"\bbook\b", "", original, flags=re.IGNORECASE).strip()
    vin, cleaned = extract_first(VIN_REGEX, cleaned)
    email, cleaned = extract_first(EMAIL_REGEX, cleaned)
    phone, cleaned = extract_first(PHONE_REGEX, cleaned)
    preferred_dt = None
    dt_text = None
    try:
        preferred_dt = date_parser.parse(cleaned, fuzzy=True, default=datetime.now(LOCAL_TZ))
        if preferred_dt.tzinfo is None:
            preferred_dt = preferred_dt.replace(tzinfo=LOCAL_TZ)
        dt_text = preferred_dt.isoformat()
    except Exception:
        preferred_dt = None
    leftover = [t for t in re.split(r"[,\n]+|\s{2,}", cleaned) if t.strip()]
    name = None
    vehicle = None
    if leftover:
        if len(leftover) <= 3:
            name = " ".join(leftover).strip()
        else:
            name = " ".join(leftover[:3]).strip()
            vehicle = " ".join(leftover[3:]).strip()
    return {
        "raw": original, "name": name, "phone": phone, "email": email,
        "vehicle": vehicle, "vin": vin, "preferred_dt": preferred_dt,
        "preferred_dt_text": dt_text,
    }

def create_calendar_event(shop, title, description, start_time=None, end_time=None):
    if not calendar_service or not shop.calendar_id:
        return
    try:
        if start_time is None:
            start_time = datetime.now(LOCAL_TZ)
        if end_time is None:
            end_time = start_time + timedelta(minutes=30)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=LOCAL_TZ)
        else:
            start_time = start_time.astimezone(LOCAL_TZ)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=LOCAL_TZ)
        else:
            end_time = end_time.astimezone(LOCAL_TZ)
        event_body = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_time.isoformat(), "timeZone": DEFAULT_TZ},
            "end": {"dateTime": end_time.isoformat(), "timeZone": DEFAULT_TZ},
        }
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event_body
        ).execute()
    except Exception as e:
        print("Calendar error:", e)

def log_estimate_lead(shop, from_number, result):
    title = f"Lead (Estimate only) – {shop.name}"
    description = (
        f"Phone: {from_number}\n"
        f"Severity: {result.get('severity')}\n"
        f"Estimate: ${result.get('estimated_cost_min')} – ${result.get('estimated_cost_max')}\n"
        f"Areas: {', '.join(result.get('damaged_areas', []))}\n"
        f"Types: {', '.join(result.get('damage_types', []))}\n\n"
        f"Summary:\n{result.get('summary')}\n\n"
        "Appointment: NOT BOOKED YET\n"
    )
    imgs = result.get("image_data_urls") or result.get("image_urls") or []
    if imgs:
        description += "\nCustomer Photos:\n"
        for src in imgs:
            description += f'\n<img src="{src}" style="width:100%;max-width:450px;margin-top:10px;">\n'
    create_calendar_event(shop, title, description)

def log_booking_event(shop, from_number, booking, last_estimate):
    name = booking.get("name") or "Unknown name"
    vehicle = booking.get("vehicle") or "Vehicle not specified"
    vin = booking.get("vin")
    email = booking.get("email")
    phone = booking.get("phone") or from_number

    title = f"Appointment – {shop.name} – {name}"
    lines = [
        f"Name: {name}", f"Phone: {phone}", f"Email: {email or 'N/A'}",
        f"Vehicle: {vehicle}", f"VIN: {vin or 'N/A'}", "",
        f"Original SMS: {booking.get('raw')}", ""
    ]
    imgs = []
    if last_estimate:
        lines += [
            "AI Estimate:",
            f"Severity: {last_estimate.get('severity')} | Estimate: ${last_estimate.get('estimated_cost_min')} – ${last_estimate.get('estimated_cost_max')}",
            f"Areas: {', '.join(last_estimate.get('damaged_areas', []))}",
            f"Types: {', '.join(last_estimate.get('damage_types', []))}",
            "",
            last_estimate.get("summary", ""), ""
        ]
        imgs = last_estimate.get("image_data_urls") or last_estimate.get("image_urls") or []
    if imgs:
        lines.append("Customer Photos:")
        for src in imgs:
            lines.append(f'\n<img src="{src}" style="width:100%;max-width:450px;margin-top:10px;">\n')
    description = "\n".join(lines)
    start_dt = booking.get("preferred_dt")
    if start_dt:
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=LOCAL_TZ)
        else:
            start_dt = start_dt.astimezone(LOCAL_TZ)
    create_calendar_event(shop, title, description, start_dt)

LAST_ESTIMATE_BY_PHONE = {}

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        return Response("Unknown shop token", status_code=403)
    form = await request.form()
    num_media = int(form.get("NumMedia", "0"))
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or "Unknown"
    reply = MessagingResponse()

    if num_media == 0 and re.search(r"\bbook\b", body, flags=re.IGNORECASE):
        booking = parse_booking_message(body)
        last_estimate = LAST_ESTIMATE_BY_PHONE.get(from_number)
        log_booking_event(shop, from_number, booking, last_estimate)
        msg = [f"✅ Thanks {booking.get('name') or ''}! Request sent to {shop.name}."]
        if booking.get("preferred_dt"):
            human_dt = booking["preferred_dt"].astimezone(LOCAL_TZ).strftime("%A, %b %d at %I:%M %p")
            msg.append(f"📅 Requested time: {human_dt}")
        else:
            msg.append("📅 Date/time unclear. The shop will contact you.")
        msg.append("If anything is wrong, text updated details starting with BOOK.")
        reply.message("\n".join(msg))
        return Response(str(reply), media_type="application/xml")

    if num_media == 0:
        if body.strip() == "1":
            reply.message(
                f"Thanks! {shop.name} will review.\nTo book: BOOK + name, phone, email, car, time."
            )
            return Response(str(reply), media_type="application/xml")
        if body.strip() == "2":
            reply.message("No problem — send 1–3 new photos.")
            return Response(str(reply), media_type="application/xml")
        reply.message(
            f"📸 Welcome to {shop.name}!\nSend 1–3 damage photos.\nThen book using BOOK + details."
        )
        return Response(str(reply), media_type="application/xml")

    images = []
    image_urls = []
    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if url:
            image_urls.append(url)
            try:
                images.append(await download_twilio_image(url))
            except Exception as e:
                print("Image download error:", e)

    embedded_data_urls = []
    for img in images:
        try:
            b64 = base64.b64encode(img).decode()
            embedded_data_urls.append(f"data:image/jpg;base64,{b64}")
        except Exception as e:
            print("Data URL encode error:", e)

    reply.message("📸 Photos received. Analyzing...")
    result = await analyze_damage(images)
    if result is None:
        reply.message("⚠️ Error analyzing image. Please resend.")
        reply.message("To book: BOOK + name, phone, email, car, time.")
        return Response(str(reply), media_type="application/xml")

    result["image_urls"] = image_urls
    result["image_data_urls"] = embedded_data_urls

    LAST_ESTIMATE_BY_PHONE[from_number] = result
    log_estimate_lead(shop, from_number, result)

    estimate_text = (
        f"🛠 AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {result['severity']}\n"
        f"Estimated Cost: ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n"
        f"Areas: {', '.join(result['damaged_areas'])}\n"
        f"Damage Types: {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n\n"
        "Reply 1 to confirm accuracy, or 2 to send more photos.\n\n"
        "To book:\nBOOK + name, phone, email, make&model, preferred time.\n"
        "Example:\nBOOK John Doe 416-555-1234 john@email.com "
        "2019 Camry this Friday 6pm VIN 1HGCM82633A123456"
    )
    reply.message(estimate_text)
    return Response(str(reply), media_type="application/xml")
