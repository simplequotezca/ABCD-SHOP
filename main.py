import os
import json
import re
import base64
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from dateutil import parser as date_parser

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Timezone handling
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Fallback handled below

# ============================================================
# FastAPI + OpenAI setup
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "America/Toronto")

# Local timezone object (used everywhere for bookings/events)
if ZoneInfo is not None:
    try:
        LOCAL_TZ = ZoneInfo(DEFAULT_TZ)
    except Exception:
        print(f"Invalid DEFAULT_TIMEZONE '{DEFAULT_TZ}', falling back to UTC")
        LOCAL_TZ = timezone.utc
else:
    LOCAL_TZ = timezone.utc

# ============================================================
# Static hosting for uploaded damage photos (Railway)
# ============================================================

# Your Railway public URL (for building full image links)
BASE_PUBLIC_URL = "https://web-production-01391.up.railway.app"

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded images at /uploads/{filename}
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def save_image_and_get_public_url(image_bytes: bytes) -> Optional[str]:
    """
    Save image bytes to /tmp/uploads and return a public HTTPS URL
    that Google Calendar can render.
    """
    try:
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        return f"{BASE_PUBLIC_URL}/uploads/{filename}"
    except Exception as e:
        print("Error saving image to disk:", e)
        return None

# ============================================================
# Multi-shop config
# ============================================================

class ShopConfig:
    def __init__(self, id: str, name: str, webhook_token: str,
                 calendar_id: Optional[str] = None):
        self.id = id
        self.name = name
        self.webhook_token = webhook_token
        self.calendar_id = calendar_id


def load_shops() -> Dict[str, ShopConfig]:
    """
    SHOPS_JSON env example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123",
        "calendar_id": "shiran.bookings@gmail.com"
      }
    ]
    """
    shops_by_token: Dict[str, ShopConfig] = {}

    raw = os.getenv("SHOPS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            for s in data:
                shops_by_token[s["webhook_token"]] = ShopConfig(
                    id=s.get("id", s["webhook_token"]),
                    name=s.get("name", "Collision Centre"),
                    webhook_token=s["webhook_token"],
                    calendar_id=s.get("calendar_id")
                )
        except Exception as e:
            print("Error parsing SHOPS_JSON:", e)

    # Fallback single shop if nothing configured
    if not shops_by_token:
        print("No SHOPS_JSON found – using default Mississauga shop.")
        shops_by_token["shop_miss_123"] = ShopConfig(
            id="miss",
            name="Mississauga Collision Centre",
            webhook_token="shop_miss_123",
            calendar_id=os.getenv("DEFAULT_CALENDAR_ID")  # optional
        )
    return shops_by_token


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# Google Calendar setup (service account JSON in env)
# ============================================================

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

# ============================================================
# Helpers: Twilio image download
# ============================================================

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

# ============================================================
# OpenAI damage analysis
# ============================================================

async def analyze_damage(image_bytes_list: List[bytes]) -> Optional[Dict[str, Any]]:
    # Tightened system prompt for better severity + cost + POV accuracy
    system_prompt = (
        "You are a professional auto-body damage estimator in Ontario, Canada (2025 prices).\n"
        "You ONLY return STRICT JSON, no extra text.\n\n"
        "ALWAYS follow these rules:\n"
        "- Use the DRIVER'S POINT OF VIEW for left/right (driver sitting in the car).\n"
        "- Carefully inspect bumper, hood, fenders, lights, trunk, doors, roof, glass, wheels, "
        "  suspension, frame, and any visible underbody or structural components.\n"
        "- If there is any sign of suspension damage, wheel pushed back, frame deformation, "
        "  airbags deployed, or major crush to a corner or front/rear, classify as 'severe'.\n"
        "- If damage is mostly cosmetic panels (dents, scrapes, small cracks) with no obvious "
        "  structural or suspension involvement, use 'minor' or 'moderate' depending on size.\n"
        "- When in doubt between two severities, choose the MORE SEVERE option.\n"
        "- Use realistic Ontario 2025 CAD repair ranges (labor, materials, paint, parts):\n"
        "    * minor: typically ~$350–$1,800\n"
        "    * moderate: typically ~$1,800–$4,500\n"
        "    * severe: typically ~$4,000–$12,000+ (crush, multiple panels, suspension/frame etc.)\n"
        "- Ensure the estimated_cost_min and estimated_cost_max match the severity level.\n"
        "- damaged_areas must be specific (e.g. 'front bumper', 'left fender', "
        "  'right rear door', 'front suspension', 'trunk lid').\n"
        "- damage_types should describe what you see (e.g. 'deep dent', 'crushed', 'scratched', "
        "  'panel deformation', 'cracked light', 'misaligned gap').\n"
        "- summary must be 2–4 clear sentences a shop owner can quickly understand.\n\n"
        "Return JSON ONLY in the following exact structure:\n"
        "{\n"
        '  \"severity\": \"minor\" | \"moderate\" | \"severe\",\n'
        '  \"estimated_cost_min\": number,\n'
        '  \"estimated_cost_max\": number,\n'
        '  \"damaged_areas\": [\"string\", ...],\n'
        '  \"damage_types\": [\"string\", ...],\n'
        '  \"summary\": \"2–4 sentence summary\"\n'
        "}"
    )

    content_list: List[Dict[str, Any]] = [
        {"type": "text", "text": "Analyze the vehicle damage in the attached images."}
    ]

    for img_bytes in image_bytes_list:
        b64 = base64.b64encode(img_bytes).decode()
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
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

        # Remove markdown wrappers if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()

        return json.loads(raw)

    except Exception as e:
        print("AI ERROR:", e)
        return None

# ============================================================
# Booking + VIN parsing
# ============================================================

VIN_REGEX = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")  # excludes I, O, Q
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

def extract_first(regex: re.Pattern, text: str) -> (Optional[str], str):
    m = regex.search(text)
    if not m:
        return None, text
    value = m.group(1 if m.lastindex else 0).strip()
    # remove that span from text
    new_text = text[:m.start()] + " " + text[m.end():]
    return value, new_text


def parse_booking_message(body: str) -> Dict[str, Any]:
    """
    Flexible booking parser:

    Works with:
      - BOOK John Doe 416... john@email 2017 Honda Civic this Friday 6pm VIN ...
      - Sun 3:00pm
      - Sunday 3pm
      - This Sunday 3:00pm
      - Dec.6th 3pm
      - 12/6 3pm
      - Any mixed order.
    """
    original = body.strip()
    # remove the word BOOK (any case)
    cleaned = re.sub(r"\bbook\b", "", original, flags=re.IGNORECASE).strip()

    vin, cleaned = extract_first(VIN_REGEX, cleaned)
    email, cleaned = extract_first(EMAIL_REGEX, cleaned)
    phone, cleaned = extract_first(PHONE_REGEX, cleaned)

    # Date/time – let dateutil guess, using local timezone as reference
    preferred_dt = None
    dt_text = None
    try:
        preferred_dt = date_parser.parse(
            cleaned,
            fuzzy=True,
            default=datetime.now(LOCAL_TZ)
        )
        # If parser returned naive, treat it as local time
        if preferred_dt.tzinfo is None:
            preferred_dt = preferred_dt.replace(tzinfo=LOCAL_TZ)
        dt_text = preferred_dt.isoformat()
    except Exception:
        preferred_dt = None

    # Remaining text -> name + vehicle guess
    leftover_tokens = [t for t in re.split(r"[,\n]+|\s{2,}", cleaned) if t.strip()]
    name = None
    vehicle = None

    if leftover_tokens:
        # simple heuristic: first 1–3 tokens = name, rest = vehicle
        if len(leftover_tokens) <= 3:
            name = " ".join(leftover_tokens).strip()
            vehicle = None
        else:
            name = " ".join(leftover_tokens[:3]).strip()
            vehicle = " ".join(leftover_tokens[3:]).strip()

    return {
        "raw": original,
        "name": name,
        "phone": phone,
        "email": email,
        "vehicle": vehicle,
        "vin": vin,
        "preferred_dt": preferred_dt,
        "preferred_dt_text": dt_text,
    }

# ============================================================
# Calendar event creation
# ============================================================

def create_calendar_event(
    shop: ShopConfig,
    title: str,
    description: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    if not calendar_service:
        print("No calendar service – skip event.")
        return
    if not shop.calendar_id:
        print(f"No calendar_id for shop {shop.name} – skip event.")
        return

    try:
        # Default event time: now (local) for 30 minutes
        if start_time is None:
            start_time = datetime.now(LOCAL_TZ)
        if end_time is None:
            end_time = start_time + timedelta(minutes=30)

        # Normalize times into LOCAL_TZ
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
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": DEFAULT_TZ,
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": DEFAULT_TZ,
            },
        }

        created = calendar_service.events().insert(
            calendarId=shop.calendar_id,
            body=event_body
        ).execute()
        print("Calendar event created:", created.get("id"))
    except Exception as e:
        print("Lead event error:", e)

def log_estimate_lead(shop: ShopConfig, from_number: str, result: Dict[str, Any]):
    """
    Called after every successful AI estimate, even if they never book.
    This gives the shop a list of all leads in Calendar.
    """
    title = f"Lead (Estimate only) – {shop.name}"
    description = (
        f"Phone: {from_number}\n"
        f"Severity: {result.get('severity')}\n"
        f"Estimate: ${result.get('estimated_cost_min')} – ${result.get('estimated_cost_max')}\n"
        f"Areas: {', '.join(result.get('damaged_areas', []))}\n"
        f"Types: {', '.join(result.get('damage_types', []))}\n\n"
        f"Summary:\n{result.get('summary')}\n\n"
        "Appointment: NOT BOOKED YET"
    )

    image_sources: List[str] = result.get("image_urls") or []

    if image_sources:
        description += "\n\nCustomer Photos:\n"
        for src in image_sources:
            description += (
                f"\n<img src=\"{src}\" style=\"width:100%;max-width:450px;margin-top:10px;\"><br>"
                f"\n{src}\n"
            )

    create_calendar_event(shop, title, description)

def log_booking_event(shop: ShopConfig, from_number: str,
                      booking: Dict[str, Any], last_estimate: Optional[Dict[str, Any]]):
    """
    Create a proper appointment event with parsed booking info.
    """
    name = booking.get("name") or "Unknown name"
    vehicle = booking.get("vehicle") or "Vehicle not specified"
    vin = booking.get("vin")
    email = booking.get("email")
    phone = booking.get("phone") or from_number

    title = f"Appointment – {shop.name} – {name}"

    lines = [
        f"Name: {name}",
        f"Phone: {phone}",
        f"Email: {email or 'N/A'}",
        f"Vehicle: {vehicle}",
        f"VIN: {vin or 'N/A'}",
        "",
        f"Original SMS: {booking.get('raw')}",
    ]

    image_sources: List[str] = []

    if last_estimate:
        lines.append("")
        lines.append("AI Estimate:")
        lines.append(
            f"Severity: {last_estimate.get('severity')} | "
            f"Estimate: ${last_estimate.get('estimated_cost_min')} – ${last_estimate.get('estimated_cost_max')}"
        )
        lines.append(f"Areas: {', '.join(last_estimate.get('damaged_areas', []))}")
        lines.append(f"Types: {', '.join(last_estimate.get('damage_types', []))}")
        lines.append("")
        lines.append(last_estimate.get("summary", ""))

        image_sources = last_estimate.get("image_urls") or []

    if image_sources:
        lines.append("")
        lines.append("Customer Photos:")
        for src in image_sources:
            lines.append(
                f'<img src="{src}" style="width:100%;max-width:450px;margin-top:10px;">'
            )
            lines.append(src)

    description = "\n".join(lines)

    start_dt = booking.get("preferred_dt")
    if start_dt:
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=LOCAL_TZ)
        else:
            start_dt = start_dt.astimezone(LOCAL_TZ)

    create_calendar_event(shop, title, description, start_dt)

# ============================================================
# Simple in-memory store of last estimate per phone
# (so we can attach estimate to booking event)
# ============================================================

LAST_ESTIMATE_BY_PHONE: Dict[str, Dict[str, Any]] = {}

# ============================================================
# SMS Webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Twilio URL: /sms-webhook?token=shop_miss_123
    """
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        return Response("Unknown shop token", status_code=403)

    form = await request.form()
    num_media = int(form.get("NumMedia", "0"))
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or "Unknown"

    reply = MessagingResponse()

    # --------------------------------------------------------
    # 1) Handle booking messages (no images, contains 'BOOK')
    # --------------------------------------------------------
    if num_media == 0 and re.search(r"\bbook\b", body, flags=re.IGNORECASE):
        booking = parse_booking_message(body)

        # Attach last estimate if we have one for this phone
        last_estimate = LAST_ESTIMATE_BY_PHONE.get(from_number)

        # Log appointment into Google Calendar (if configured)
        log_booking_event(shop, from_number, booking, last_estimate)

        confirm_lines = [
            f"✅ Thanks {booking.get('name') or ''}! Your request has been sent to {shop.name}.",
        ]

        if booking.get("preferred_dt"):
            human_dt = booking["preferred_dt"].astimezone(LOCAL_TZ).strftime(
                "%A, %b %d at %I:%M %p"
            )
            confirm_lines.append(f"📅 Requested time: {human_dt}")
        else:
            confirm_lines.append(
                "📅 We couldn't clearly detect a date/time, but the shop will text or call you to confirm a slot."
            )

        confirm_lines.append(
            "If anything is wrong, just text updated details starting with the word BOOK."
        )

        reply.message("\n".join(confirm_lines))
        return Response(str(reply), media_type="application/xml")

    # --------------------------------------------------------
    # 2) If no media and not a booking -> welcome / simple flow
    # --------------------------------------------------------
    if num_media == 0:
        # Optionally handle quick replies 1 / 2
        if body.strip() == "1":
            reply.message(
                f"✅ Thanks! {shop.name} will review your estimate and follow up shortly.\n\n"
                "To book now, text:\n"
                "BOOK + your name, phone #, email, make & model of your car, "
                "and preferred date/time (any order). You can skip details you don't know."
            )
            return Response(str(reply), media_type="application/xml")
        elif body.strip() == "2":
            reply.message(
                "No problem! Please send 1–3 new photos from different angles so our AI can re-check the damage."
            )
            return Response(str(reply), media_type="application/xml")

        # Default welcome
        reply.message(
            f"📸 Welcome to {shop.name}!\n\n"
            "To get a fast AI damage estimate, please send 1–3 clear photos of the vehicle damage.\n\n"
            "After you get your estimate, you can book by texting:\n"
            "BOOK + your name, phone #, email, make & model of your car, "
            "and preferred date/time (any order). You can include as much or as little as you want."
        )
        return Response(str(reply), media_type="application/xml")

    # --------------------------------------------------------
    # 3) We have images -> run AI pipeline
    # --------------------------------------------------------

    images: List[bytes] = []
    public_image_urls: List[str] = []

    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if not url:
            continue
        try:
            img = await download_twilio_image(url)
            images.append(img)

            public_url = save_image_and_get_public_url(img)
            if public_url:
                public_image_urls.append(public_url)
        except Exception as e:
            print("Image download/save error:", e)

    # Acknowledge receipt
    reply.message(
        "📸 Thanks! We received your photos.\n"
        "Analyzing damage now..."
    )

    # Run AI estimator
    result = await analyze_damage(images)

    if result is None:
        reply.message(
            "⚠️ Error analyzing image. Please resend your photo."
        )
        reply.message(
            "To book an appointment, reply:\n"
            "BOOK + your name, phone #, email, make & model of your car, "
            "and preferred date/time (any order)."
        )
        return Response(str(reply), media_type="application/xml")

    # Attach public image URLs so calendar events can embed photos
    result["image_urls"] = public_image_urls

    # Store last estimate per phone for later bookings
    LAST_ESTIMATE_BY_PHONE[from_number] = result

    # Log a lead into calendar even if they never book
    log_estimate_lead(shop, from_number, result)

    # Build human-friendly SMS
    estimate_text = (
        f"🛠 AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {result['severity']}\n"
        f"Estimated Cost: ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n"
        f"Areas: {', '.join(result['damaged_areas'])}\n"
        f"Damage Types: {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n\n"
        "Reply 1 to confirm this looks accurate, or 2 to send more photos.\n\n"
        "To book an appointment, reply:\n"
        "BOOK + your name, phone #, email, make & model of your car, "
        "and preferred date/time (any order).\n"
        "Example:\n"
        "BOOK John Doe 416-555-1234 john@email.com "
        "2018 Toyota Camry this Friday 6pm VIN 1HGCM82633A123456"
    )

    reply.message(estimate_text)
    return Response(str(reply), media_type="application/xml")
