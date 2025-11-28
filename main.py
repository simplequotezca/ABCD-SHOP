import os
import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

# Google Calendar imports (fail-soft if not configured)
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover - only if google libs missing
    Credentials = None
    build = None

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("uvicorn.error")

# ============================================================
# Environment
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required")

SHOPS_JSON = os.getenv("SHOPS_JSON")
if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON env var is required")

# Optional: Google service account JSON (stringified)
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Shop configuration
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    hours: Optional[Dict[str, Any]] = None


def load_shops() -> Dict[str, Shop]:
    try:
        raw = json.loads(SHOPS_JSON)
        shops: List[Shop] = [Shop.model_validate(s) for s in raw]
    except Exception as e:
        logger.exception("Failed to parse SHOPS_JSON: %s", e)
        raise RuntimeError("Invalid SHOPS_JSON format") from e

    by_token: Dict[str, Shop] = {}
    for shop in shops:
        if shop.webhook_token in by_token:
            raise RuntimeError(f"Duplicate webhook_token in SHOPS_JSON: {shop.webhook_token}")
        by_token[shop.webhook_token] = shop
    return by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()

# In-memory booking state (per phone number)
PENDING_BOOKINGS: Dict[str, Dict[str, Any]] = {}

# ============================================================
# Google Calendar helpers
# ============================================================

def get_calendar_service():
    """Return a Calendar API client, or None if misconfigured."""
    if not (GOOGLE_SERVICE_ACCOUNT_JSON and Credentials and build):
        logger.warning("Google Calendar not configured; skipping event creation")
        return None

    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        scopes = ["https://www.googleapis.com/auth/calendar"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        logger.exception("Failed to init Google Calendar service: %s", e)
        return None


def create_calendar_event(shop: Shop, from_number: str, details: str) -> Optional[str]:
    """
    Very simple booking: create a 30-minute event for 'tomorrow at 10:00'
    in the shop's calendar. The full 'details' text goes into description.
    Returns event HTML link or None.
    """
    if not shop.calendar_id:
        logger.warning("Shop %s has no calendar_id; skipping event", shop.id)
        return None

    service = get_calendar_service()
    if not service:
        return None

    try:
        # Naive time: tomorrow at 10:00 local server time
        start_dt = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(minutes=30)

        event_body = {
            "summary": f"AI Damage Estimate Booking - {from_number}",
            "description": details,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Toronto"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Toronto"},
        }

        created = service.events().insert(calendarId=shop.calendar_id, body=event_body).execute()
        html_link = created.get("htmlLink")
        logger.info("Created calendar event for shop %s: %s", shop.id, html_link)
        return html_link
    except Exception as e:
        logger.exception("Error creating Calendar event: %s", e)
        return None

# ============================================================
# Twilio media download (Option B)
# ============================================================

def download_twilio_image(url: str) -> Optional[bytes]:
    """
    Download a Twilio-hosted image using Account SID/Auth.
    This avoids OpenAI's `invalid_image_url` problem.
    """
    try:
        logger.info("Downloading Twilio media: %s", url)
        resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.exception("Failed to download Twilio media: %s", e)
        return None

# ============================================================
# AI damage estimator (image + text)
# ============================================================

DAMAGE_SYSTEM_PROMPT = (
    "You are an expert auto body damage estimator in Ontario, Canada (2025). "
    "Analyze the vehicle photos and the customer's text. "
    "Output a clear, customer-friendly estimate with:\n"
    "1) Severity category: Minor / Moderate / Severe\n"
    "2) Main areas affected (bumper, hood, fenders, doors, etc.)\n"
    "3) Key damage types (scratches, dents, cracks, misalignment, structural risk)\n"
    "4) Rough total cost range in CAD using typical Ontario 2025 rates\n"
    "5) A short note that this is a visual, preliminary estimate and not a final bill.\n"
    "Keep it under 250 words."
)


def run_damage_estimator(shop: Shop, user_text: str, image_bytes_list: List[bytes]) -> str:
    """
    Send images + text to OpenAI using base64 data URLs (Option B).
    """
    content: List[Dict[str, Any]] = []

    intro = (
        f"Shop: {shop.name}\n"
        f"Customer message: {user_text or '(no extra details)'}\n\n"
        "Use the attached images plus this message to produce the estimate."
    )
    content.append({"type": "text", "text": intro})

    for idx, img_bytes in enumerate(image_bytes_list):
        if not img_bytes:
            continue
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"
        content.append(
            {
                "type": "input_image",
                "image_url": {"url": data_url},
            }
        )

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",  # vision-capable
            messages=[
                {"role": "system", "content": DAMAGE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=700,
        )
        reply = completion.choices[0].message.content or ""
        reply = reply.strip()
        logger.info("AI damage estimate generated (%d chars)", len(reply))
        return reply
    except Exception as e:
        logger.exception("Error during AI damage estimation: %s", e)
        return (
            "Sorry, our AI estimator had an issue analyzing the photos this time.\n\n"
            "Please reply with a brief description of the damage, and the shop can follow up manually."
        )

# ============================================================
# Conversation logic (text only + booking flow)
# ============================================================

def handle_text_only(shop: Shop, from_number: str, body: str) -> str:
    """
    Handles flows when there are no images:
    - BOOK keyword -> booking dialogue
    - Pending booking details
    - General first-contact message
    """
    lower = (body or "").strip().lower()

    # Existing booking flow
    if from_number in PENDING_BOOKINGS:
        # Treat this message as details for the booking
        details = body.strip()
        state = PENDING_BOOKINGS.pop(from_number, None)

        details_full = (
            f"Shop: {shop.name}\n"
            f"Customer phone: {from_number}\n"
            f"Booking details (from customer):\n{details}"
        )

        link = create_calendar_event(shop, from_number, details_full)
        if link:
            return (
                "Thanks! We've booked a FREE in-person inspection for you.\n\n"
                "The shop will review and confirm the exact time shortly.\n"
                "You can view the internal booking in their calendar."
            )
        else:
            return (
                "Thanks! We've recorded your details and sent them to the shop.\n"
                "They will contact you shortly to confirm an exact appointment time."
            )

    # New BOOK keyword
    if lower == "book":
        PENDING_BOOKINGS[from_number] = {"stage": "await_details", "shop_id": shop.id}
        return (
            "Great! To book your FREE in-person inspection, please reply with:\n"
            "• Your full name\n"
            "• Email\n"
            "• Vehicle year / make / model\n"
            "• Preferred day & time\n\n"
            "You can type it all in one message."
        )

    # Generic intro / help
    if not body or lower in {"hi", "hello", "hey"}:
        return (
            f"Hi there! Thanks for reaching out to {shop.name}.\n\n"
            "You can:\n"
            "• Text photos of the damage for a fast AI-powered estimate, or\n"
            "• Reply with BOOK to schedule a FREE in-person inspection."
        )

    # Fallback text-only response
    return (
        f"Thanks for your message to {shop.name}.\n\n"
        "To get the most accurate AI estimate, please reply with clear photos of the damage "
        "from a few angles (front, side, close-up). "
        "Or reply with BOOK to schedule a FREE in-person inspection."
    )

# ============================================================
# FastAPI app + routes
# ============================================================

app = FastAPI()


@app.get("/", response_class=PlainTextResponse)
async def root():
    return PlainTextResponse("OK")  # health-check


@app.post("/sms-webhook", response_class=PlainTextResponse)
async def sms_webhook(request: Request, token: str = Query(...)):
    """
    Twilio SMS/MMS webhook.
    Must always return valid TwiML as XML.
    """
    resp = MessagingResponse()

    try:
        shop = SHOPS_BY_TOKEN.get(token)
        if not shop:
            logger.error("Unknown webhook token: %s", token)
            resp.message("Configuration error: unknown shop destination.")
            return PlainTextResponse(str(resp), media_type="application/xml")

        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From") or "Unknown"
        num_media = int(form.get("NumMedia") or "0")

        logger.info(
            "Incoming SMS for shop=%s from=%s, body=%r, images=%d",
            shop.id, from_number, body, num_media
        )

        reply_text: str

        if num_media > 0:
            # Download all Twilio images first (Option B)
            image_bytes_list: List[bytes] = []
            for i in range(num_media):
                media_url = form.get(f"MediaUrl{i}")
                if not media_url:
                    continue
                img_bytes = download_twilio_image(media_url)
                if img_bytes:
                    image_bytes_list.append(img_bytes)

            if not image_bytes_list:
                # If download failed, fall back gracefully
                logger.warning("No images could be downloaded; falling back to text-only flow")
                reply_text = (
                    "We had trouble downloading your photos from the network.\n\n"
                    "Please try sending them again, or reply with BOOK to schedule a "
                    "FREE in-person inspection."
                )
            else:
                estimate = run_damage_estimator(shop, body, image_bytes_list)
                reply_text = (
                    f"AI Damage Estimate for {shop.name}:\n\n"
                    f"{estimate}\n\n"
                    "Reply with BOOK to schedule a FREE in-person inspection."
                )
        else:
            # No images → handle text-only logic (greeting, BOOK, etc.)
            reply_text = handle_text_only(shop, from_number, body)

        resp.message(reply_text)

        twiml = str(resp)
        logger.info("Replying to Twilio with TwiML: %r", twiml)
        return PlainTextResponse(twiml, media_type="application/xml")

    except Exception as e:
        logger.exception("Error in /sms-webhook handler: %s", e)
        resp = MessagingResponse()
        resp.message(
            "Sorry, something went wrong on our end while processing your message.\n"
            "Please try again in a few minutes, or call the shop directly."
        )
        return PlainTextResponse(str(resp), media_type="application/xml")
