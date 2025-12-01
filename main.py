import os
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# ---------------------------------------------------------
# Optional Google Calendar imports (fail gracefully)
# ---------------------------------------------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:  # requirements not yet updated
    GOOGLE_CALENDAR_AVAILABLE = False

try:
    from dateutil import parser as date_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

# ---------------------------------------------------------
# FastAPI + OpenAI setup
# ---------------------------------------------------------

app = FastAPI()
client = OpenAI()

OPENAI_MODEL = "gpt-4o-mini"
TIMEZONE = os.getenv("DEFAULT_TZ", "America/Toronto")

# ---------------------------------------------------------
# Multi-shop config via SHOPS_JSON
# ---------------------------------------------------------


class ShopConfig:
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id")
        self.name: str = data.get("name", "Your Body Shop")
        self.webhook_token: str = data.get("webhook_token")
        self.calendar_id: Optional[str] = data.get("calendar_id")


def load_shops() -> Dict[str, ShopConfig]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError(
            "SHOPS_JSON env var is required. Example:\n"
            '[{"id":"miss","name":"Mississauga Collision","webhook_token":"shop_miss_123",'
            '"calendar_id":"your-calendar-id@group.calendar.google.com"}]'
        )
    try:
        data = json.loads(raw)
        shops: Dict[str, ShopConfig] = {}
        for item in data:
            shop = ShopConfig(item)
            if not shop.webhook_token:
                continue
            shops[shop.webhook_token] = shop
        return shops
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid SHOPS_JSON: {e}") from e


SHOPS_BY_TOKEN = load_shops()

# ---------------------------------------------------------
# In-memory cache of last estimates by phone (for booking)
# ---------------------------------------------------------

# key = phone number, value = dict with estimate_text, shop_id, image_urls, timestamp
LAST_ESTIMATES: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------
# Google Calendar helpers
# ---------------------------------------------------------


def get_calendar_service():
    """Builds a Google Calendar API service using a service account.

    Requires:
      - GOOGLE_SERVICE_ACCOUNT_JSON env var (full JSON string)
      - google-api-python-client & google-auth installed
    """
    if not GOOGLE_CALENDAR_AVAILABLE:
        print("[WARN] Google Calendar libraries not installed. Calendar features disabled.")
        return None

    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        print("[WARN] GOOGLE_SERVICE_ACCOUNT_JSON not set. Calendar features disabled.")
        return None

    try:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        print(f"[ERROR] Failed to build Google Calendar service: {e}")
        return None


def calendar_insert_event(
    shop: ShopConfig,
    summary: str,
    description: str,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    all_day: bool = False,
):
    """Insert an event into the shop's calendar. Safe-fail."""
    if not shop.calendar_id:
        print(f"[WARN] Shop {shop.id} has no calendar_id configured.")
        return

    service = get_calendar_service()
    if not service:
        return

    try:
        if all_day:
            start_date = (start_dt or datetime.now()).date()
            end_date = start_date + timedelta(days=1)
            event = {
                "summary": summary,
                "description": description,
                "start": {"date": start_date.isoformat(), "timeZone": TIMEZONE},
                "end": {"date": end_date.isoformat(), "timeZone": TIMEZONE},
            }
        else:
            if not start_dt:
                start_dt = datetime.now()
            if not end_dt:
                end_dt = start_dt + timedelta(hours=2)
            event = {
                "summary": summary,
                "description": description,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": TIMEZONE,
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": TIMEZONE,
                },
            }

        created = service.events().insert(
            calendarId=shop.calendar_id,
            body=event,
        ).execute()
        print(f"[INFO] Calendar event created: {created.get('id')}")
    except Exception as e:
        print(f"[ERROR] Failed to insert calendar event: {e}")


def create_lead_event(
    shop: ShopConfig,
    from_number: str,
    estimate_text: str,
    image_urls: List[str],
):
    """Create an all-day 'lead' event so shop can follow up even if no booking."""
    desc_lines = [
        f"Phone: {from_number}",
        "",
        "AI Estimate:",
        estimate_text,
    ]
    if image_urls:
        desc_lines.append("")
        desc_lines.append("Photos:")
        desc_lines.extend(image_urls)

    description = "\n".join(desc_lines)
    summary = f"Lead – Pending Booking – {from_number}"
    calendar_insert_event(shop, summary, description, all_day=True)


def create_booking_event(
    shop: ShopConfig,
    from_number: str,
    customer: Dict[str, str],
    estimate_text: Optional[str],
    image_urls: List[str],
):
    """Create a timed appointment event based on customer details."""
    name = customer.get("name") or from_number
    phone = customer.get("phone") or from_number
    email = customer.get("email") or "N/A"
    preferred = customer.get("preferred") or "Not specified"
    vehicle = customer.get("vehicle") or "Not specified"

    lines = [
        f"Customer Name: {name}",
        f"Phone: {phone}",
        f"Email: {email}",
        f"Preferred Date & Time (raw): {preferred}",
        "",
        "Vehicle Info:",
        vehicle,
        "",
        "AI Estimate Details:",
        estimate_text or "(Estimate not found in memory – server may have restarted.)",
    ]
    if image_urls:
        lines.append("")
        lines.append("Photos:")
        lines.extend(image_urls)

    description = "\n".join(lines)
    summary = f"AI Estimate Booking – {name}"

    # Try to parse preferred time, but if it fails, fall back to all-day today.
    start_dt = None
    end_dt = None
    all_day = False

    if DATEUTIL_AVAILABLE and preferred and preferred.lower() != "not specified":
        try:
            parsed = date_parser.parse(preferred, fuzzy=True)
            start_dt = parsed
            end_dt = parsed + timedelta(hours=2)
        except Exception as e:
            print(f"[WARN] Could not parse preferred date/time '{preferred}': {e}")
            all_day = True
    else:
        all_day = True

    calendar_insert_event(
        shop,
        summary,
        description,
        start_dt=start_dt,
        end_dt=end_dt,
        all_day=all_day,
    )


# ---------------------------------------------------------
# Twilio + OpenAI helper functions
# ---------------------------------------------------------


async def download_image_as_data_url(url: str) -> Optional[str]:
    """Downloads image from Twilio CDN and returns a data: URL base64 string."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client_http:
            resp = await client_http.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return f"data:{content_type};base64,{b64}"
    except Exception as e:
        print(f"[ERROR] Failed to download image {url}: {e}")
        return None


async def generate_damage_estimate(
    image_data_urls: List[str],
    shop_name: str,
) -> str:
    """Calls OpenAI vision model to generate the AI damage estimate text."""
    # We only use the first 3 images to keep prompt size manageable
    image_contents = []
    for data_url in image_data_urls[:3]:
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    prompt_text = (
        f"You are an expert auto body damage estimator for {shop_name} in Ontario, Canada. "
        "Carefully analyze the photos and produce a clear, SMS-friendly estimate. "
        "Always respond in this exact structure:\n\n"
        "🛠 **AI Damage Estimate**\n\n"
        "**Severity:** (minor / moderate / severe)\n"
        "**Estimated Cost:** $LOW – $HIGH\n"
        "**Areas:** list all affected areas (e.g., rear bumper, trunk lid, tail light)\n"
        "**Damage Types:** list damage types (e.g., dent, crack, scratches, panel deformation)\n\n"
        "Then a short paragraph explaining the damage and what work is likely needed.\n\n"
        "End with: 'Reply *1* to confirm or *2* to upload more photos.'"
    )

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional auto body damage estimator.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        *image_contents,
                    ],
                },
            ],
        )

        # New OpenAI client: message.content is a string
        estimate_text = completion.choices[0].message.content.strip()
        return estimate_text
    except Exception as e:
        print(f"[ERROR] OpenAI vision error: {e}")
        return (
            "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
            "Please try again in a few minutes."
        )


def parse_booking_message(body: str) -> Dict[str, str]:
    """
    Parses a booking message of the form:

    BOOKING:
    Name: John Doe
    Phone: 647-555-1234
    Email: john@example.com
    Preferred Date & Time: Tue Dec 5 at 3pm
    Vehicle: 2017 Honda Odyssey blue
    """
    result = {
        "name": "",
        "phone": "",
        "email": "",
        "preferred": "",
        "vehicle": "",
    }

    text = body.replace("\r", "")
    # Remove leading 'BOOKING:' line
    if ":" in text:
        parts = text.split("BOOKING:", 1)
        if len(parts) == 2:
            text = parts[1]

    for line in text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key.startswith("name"):
            result["name"] = value
        elif key.startswith("phone"):
            result["phone"] = value
        elif key.startswith("email"):
            result["email"] = value
        elif "preferred" in key or "date" in key or "time" in key:
            result["preferred"] = value
        elif "vehicle" in key or "car" in key:
            result["vehicle"] = value

    return result


def get_shop_from_request(request: Request) -> ShopConfig:
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=403, detail="Missing shop token")
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=403, detail="Invalid shop token")
    return shop


# ---------------------------------------------------------
# Twilio webhook
# ---------------------------------------------------------


@app.post("/sms-webhook", response_class=PlainTextResponse)
async def sms_webhook(request: Request):
    """
    Main Twilio webhook.

    - If message contains media (photos): run AI estimator, send estimate,
      create a lead event in calendar, and cache estimate in memory.
    - If message is '1': send booking instructions.
    - If message is '2': ask for more photos.
    - If message starts with 'BOOKING:': parse booking details, create calendar event.
    - Otherwise: send intro / help message.
    """
    shop = get_shop_from_request(request)

    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or "Unknown"
    num_media = int(form.get("NumMedia") or "0")

    print(f"[INFO] Incoming SMS from {from_number}, media={num_media}, body='{body}'")

    resp = MessagingResponse()

    # -------------------------------------------------
    # Booking flow: BOOKING: ...
    # -------------------------------------------------
    upper_body = body.upper()
    if upper_body.startswith("BOOKING:"):
        customer = parse_booking_message(body)
        estimate_record = LAST_ESTIMATES.get(from_number, {})
        estimate_text = estimate_record.get("estimate_text")
        image_urls = estimate_record.get("image_urls", [])

        create_booking_event(shop, from_number, customer, estimate_text, image_urls)

        msg = (
            "✅ Your appointment request has been sent to "
            f"{shop.name}. They will confirm your date & time shortly.\n\n"
            "If you need to change anything, just reply here."
        )
        resp.message(msg)
        return PlainTextResponse(str(resp))

    # -------------------------------------------------
    # Simple commands: 1 = confirm, 2 = more photos
    # (Estimator logic itself is unchanged)
    # -------------------------------------------------
    if body == "1":
        # Confirmation – ask them to send BOOKING block
        msg = (
            "Great! To book your repair appointment, please reply in this format:\n\n"
            "BOOKING:\n"
            "Name: Your Name\n"
            "Phone: Your Phone Number\n"
            "Email: your@email.com\n"
            "Preferred Date & Time: e.g. Tue Dec 5 at 3pm\n"
            "Vehicle: Year Make Model (colour)\n\n"
            "We'll add this to our calendar and confirm with you."
        )
        resp.message(msg)
        return PlainTextResponse(str(resp))

    if body == "2":
        msg = (
            "No problem! Please send 1–3 more photos of the damage from different angles, "
            "and I'll refine the estimate."
        )
        resp.message(msg)
        return PlainTextResponse(str(resp))

    # -------------------------------------------------
    # Media received: run AI estimator (working logic)
    # -------------------------------------------------
    if num_media > 0:
        # Initial friendly acknowledgement (unchanged)
        intro_msg = (
            "📸 Thanks! We received your photos.\n\n"
            "Our AI estimator is reviewing the damage now — "
            "you'll get the breakdown shortly."
        )
        resp.message(intro_msg)

        # Download images from Twilio
        image_data_urls: List[str] = []
        image_urls_for_log: List[str] = []
        for i in range(num_media):
            media_url = form.get(f"MediaUrl{i}")
            if not media_url:
                continue
            image_urls_for_log.append(media_url)
            data_url = await download_image_as_data_url(media_url)
            if data_url:
                image_data_urls.append(data_url)

        if not image_data_urls:
            error_msg = (
                "⚠️ AI Processing Error: Couldn't download your photos from our servers. "
                "Please try sending them again."
            )
            resp.message(error_msg)
            return PlainTextResponse(str(resp))

        # Call OpenAI vision model (same estimator behavior)
        estimate_text = await generate_damage_estimate(image_data_urls, shop.name)

        # Send estimate to customer
        resp.message(estimate_text)

        # Cache estimate in memory for future booking
        LAST_ESTIMATES[from_number] = {
            "shop_id": shop.id,
            "timestamp": datetime.utcnow().isoformat(),
            "estimate_text": estimate_text,
            "image_urls": image_urls_for_log,
        }

        # Create a lead event so shop can follow up even if they never book
        create_lead_event(shop, from_number, estimate_text, image_urls_for_log)

        return PlainTextResponse(str(resp))

    # -------------------------------------------------
    # No media, no special command – send intro/help
    # -------------------------------------------------
    intro = (
        f"Welcome to {shop.name}!\n\n"
        "To get a fast AI damage estimate, please send 1–3 clear photos of the damage. "
        "We'll analyze them and text you a detailed estimate, plus an option to book "
        "an appointment."
    )
    resp.message(intro)
    return PlainTextResponse(str(resp))
