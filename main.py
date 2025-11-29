import os
import json
import base64
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from googleapiclient.discovery import build
from google.oauth2 import service_account

from urllib.request import Request as URLRequest, urlopen
from urllib.error import URLError, HTTPError

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto-estimator")

# ============================================================
# Environment / Config
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SHOPS_JSON = os.getenv("SHOPS_JSON")

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

if not SHOPS_JSON:
    # Minimal default config so the app can still boot for health checks
    logger.warning("SHOPS_JSON not set – using a default demo shop config.")
    SHOPS_CONFIG: List[Dict[str, Any]] = [
        {
            "id": "demo",
            "name": "Demo Collision Centre",
            "webhook_token": "demo_token_123",
            "calendar_id": None,
        }
    ]
else:
    try:
        SHOPS_CONFIG = json.loads(SHOPS_JSON)
    except json.JSONDecodeError:
        logger.exception("Failed to parse SHOPS_JSON, falling back to demo shop.")
        SHOPS_CONFIG = [
            {
                "id": "demo",
                "name": "Demo Collision Centre",
                "webhook_token": "demo_token_123",
                "calendar_id": None,
            }
        ]

# Map webhook token -> shop dict for quick lookup
SHOPS_BY_TOKEN: Dict[str, Dict[str, Any]] = {
    shop["webhook_token"]: shop for shop in SHOPS_CONFIG if "webhook_token" in shop
}

# In-memory sessions for simple booking flow (ok for demo, not for production scale)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# OpenAI client
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY is not set – AI estimator will be disabled.")

# ============================================================
# Google Calendar helpers
# ============================================================

_calendar_service: Optional[Any] = None


def get_calendar_service() -> Optional[Any]:
    """
    Lazily create and cache a Google Calendar service using a service account.
    Returns None if credentials are missing or misconfigured.
    """
    global _calendar_service

    if _calendar_service is not None:
        return _calendar_service

    creds_info: Optional[Dict[str, Any]] = None

    try:
        if GOOGLE_SERVICE_ACCOUNT_JSON:
            creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        elif GOOGLE_SERVICE_ACCOUNT_FILE and os.path.exists(GOOGLE_SERVICE_ACCOUNT_FILE):
            with open(GOOGLE_SERVICE_ACCOUNT_FILE, "r", encoding="utf-8") as f:
                creds_info = json.load(f)

        if not creds_info:
            logger.info("Google service account not configured – calendar booking disabled.")
            return None

        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        logger.info("Google Calendar service initialized.")
        return _calendar_service
    except Exception:
        logger.exception("Failed to initialize Google Calendar service.")
        _calendar_service = None
        return None


def create_calendar_booking(
    shop: Dict[str, Any],
    booking: Dict[str, Any],
    from_phone: str,
) -> bool:
    """
    Create a calendar event for the booking.
    Returns True if the event was created successfully, False otherwise.
    """
    calendar_id = shop.get("calendar_id")
    if not calendar_id:
        logger.info("Shop %s has no calendar_id configured.", shop.get("id"))
        return False

    service = get_calendar_service()
    if not service:
        return False

    start_dt: datetime = booking["datetime"]
    end_dt = start_dt + timedelta(hours=1)

    description_lines = [
        "Auto body appointment from SMS.",
        f"Shop: {shop.get('name')}",
        f"Name: {booking.get('name')}",
        f"Phone: {from_phone}",
        f"Email: {booking.get('email')}",
        f"Vehicle: {booking.get('vehicle')}",
        f"Requested time (local): {start_dt.isoformat()}",
    ]
    description = "\n".join(description_lines)

    event_body = {
        "summary": f"Collision Inspection – {booking.get('name')}",
        "description": description,
        "start": {"dateTime": start_dt.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
    }

    try:
        service.events().insert(calendarId=calendar_id, body=event_body, sendUpdates="all").execute()
        logger.info("Created calendar event for shop %s at %s", shop.get("id"), start_dt)
        return True
    except Exception:
        logger.exception("Error creating Google Calendar event.")
        return False


# ============================================================
# Image helpers
# ============================================================


def download_twilio_image_as_base64(url: str) -> Optional[str]:
    """
    Download an image from a Twilio MediaUrl using basic auth and return a base64 string.
    Returns None on failure.
    """
    if not url:
        return None

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        logger.error("Twilio credentials not set – cannot download media.")
        return None

    try:
        auth_str = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}"
        auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("ascii")

        req = URLRequest(url)
        req.add_header("Authorization", f"Basic {auth_b64}")

        with urlopen(req, timeout=15) as resp:
            data = resp.read()

        if not data:
            logger.error("No data received when downloading Twilio media.")
            return None

        img_b64 = base64.b64encode(data).decode("ascii")
        return img_b64
    except (HTTPError, URLError) as e:
        logger.exception("Network error downloading Twilio image: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error downloading Twilio image.")
        return None


# ============================================================
# AI Damage Estimator
# ============================================================


async def run_damage_estimator(shop: Dict[str, Any], media_urls: List[str]) -> str:
    """
    Call OpenAI vision model on the uploaded images and return a human-readable estimate string.
    """
    if not client:
        return (
            "Our AI estimator is temporarily unavailable. "
            "Please reply with a brief description of the damage and the shop will follow up manually."
        )

    image_contents: List[Dict[str, Any]] = []
    for url in media_urls:
        img_b64 = download_twilio_image_as_base64(url)
        if img_b64:
            image_contents.append(
                {
                    "type": "input_image",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

    if not image_contents:
        logger.warning("No valid images could be downloaded from Twilio media URLs.")
        return (
            "Sorry, our AI estimator had an issue reading the photos this time.\n\n"
            "Please reply with a brief description of the damage, and the shop can follow up manually.\n\n"
            "You can also reply with BOOK to schedule a FREE in-person inspection."
        )

    shop_name = shop.get("name", "the shop")

    system_prompt = (
        "You are an experienced auto body damage estimator in Ontario, Canada (2025). "
        "You will be shown 1–5 photos of a damaged vehicle. "
        "Your job is to:\n"
        "1) Identify which areas of the vehicle are damaged.\n"
        "2) Classify overall severity as Minor, Moderate, or Severe.\n"
        "3) Provide a realistic cost range in CAD for Ontario collision repair shops.\n"
        "4) Explain in plain language what repairs are likely required.\n"
        "5) Add a disclaimer that this is a preliminary visual estimate, not a final invoice.\n\n"
        "Use concise bullet points. Never invent unrelated damage. If photos are unclear, say so."
    )

    user_text = (
        f"Customer has texted photos of collision damage for {shop_name}.\n\n"
        "Based ONLY on these images, provide:\n"
        "- Severity (Minor / Moderate / Severe)\n"
        "- Estimated cost range in CAD\n"
        "- Key damaged areas and likely repairs\n"
        "- Any safety concerns (e.g., airbags, frame, alignment)\n"
        "- Short, friendly explanation for the customer.\n\n"
        "Respond in this format:\n"
        "Severity: <Minor/Moderate/Severe>\n"
        "Estimated Cost (Ontario 2025): $X – $Y\n"
        "Areas: <list>\n"
        "Damage Types: <list>\n\n"
        "<2–4 short bullet points explaining reasoning and next steps.>\n\n"
        "End with: 'This is a visual, preliminary estimate and not a final repair bill.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}] + image_contents,
        },
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        estimate_text = completion.choices[0].message.content.strip()
        logger.info("AI damage estimate generated successfully.")
        return estimate_text
    except Exception as e:
        logger.exception("Error during OpenAI damage estimation: %s", e)
        return (
            f"AI Damage Estimate for {shop_name}:\n\n"
            "Sorry, our AI estimator had an issue analyzing the photos this time.\n\n"
            "Please reply with a brief description of the damage, and the shop can follow up manually.\n\n"
            "You can also reply with BOOK to schedule a FREE in-person inspection."
        )


# ============================================================
# Booking Flow
# ============================================================


def start_booking_session(from_phone: str, shop: Dict[str, Any]) -> str:
    SESSIONS[from_phone] = {
        "state": "booking_name",
        "shop_id": shop.get("id"),
        "started_at": datetime.utcnow().isoformat(),
    }
    return (
        f"Great! Let's schedule a FREE in-person inspection at {shop.get('name')}.\n\n"
        "First, what's your full name?"
    )


def handle_booking_step(from_phone: str, body: str, shop: Dict[str, Any]) -> Optional[str]:
    session = SESSIONS.get(from_phone)
    if not session:
        return None

    state = session.get("state")

    if state == "booking_name":
        session["name"] = body.strip()
        session["state"] = "booking_vehicle"
        return (
            f"Thanks {session['name']}!\n\n"
            "What vehicle will you be bringing in (year, make, model, colour)?"
        )

    if state == "booking_vehicle":
        session["vehicle"] = body.strip()
        session["state"] = "booking_email"
        return "Got it. What's the best email for your booking confirmation?"

    if state == "booking_email":
        session["email"] = body.strip()
        session["state"] = "booking_datetime"
        return (
            "Perfect.\n\n"
            "Lastly, what date and time work best for you?\n"
            "Please reply in this format: YYYY-MM-DD HH:MM (for example 2025-12-01 14:30)."
        )

    if state == "booking_datetime":
        raw = body.strip()
        try:
            appt_dt = datetime.strptime(raw, "%Y-%m-%d %H:%M")
        except ValueError:
            return (
                "I couldn't read that date/time.\n"
                "Please use this format: YYYY-MM-DD HH:MM (for example 2025-12-01 14:30)."
            )

        session["datetime"] = appt_dt
        session["datetime_raw"] = raw

        success = create_calendar_booking(shop, session, from_phone)
        pretty = appt_dt.strftime("%A, %B %d at %I:%M %p")

        # Clear session
        SESSIONS.pop(from_phone, None)

        if success:
            return (
                f"You're all set! We've booked you in for {pretty} at {shop.get('name')}.\n\n"
                "You'll receive a confirmation from the shop if any changes are needed."
            )
        else:
            return (
                f"Thanks! We've recorded your preferred time ({pretty}) and details.\n\n"
                f"{shop.get('name')} will contact you to confirm your appointment."
            )

    # Unknown state – reset session
    logger.warning("Unknown booking state for %s, resetting session.", from_phone)
    SESSIONS.pop(from_phone, None)
    return (
        f"Something went wrong with your booking at {shop.get('name')}.\n"
        "Please reply with BOOK to start again."
    )


def handle_text_message(shop: Dict[str, Any], from_phone: str, body: str) -> str:
    """
    Handle non-image messages: greeting, BOOK flow, or fallback help.
    """
    body_clean = (body or "").strip()
    body_lower = body_clean.lower()

    # If user is mid-booking, advance that flow
    if from_phone in SESSIONS:
        maybe_reply = handle_booking_step(from_phone, body_clean, shop)
        if maybe_reply:
            return maybe_reply

    # Start booking flow
    if body_lower == "book" or body_lower.startswith("book "):
        return start_booking_session(from_phone, shop)

    # Default info / greeting
    return (
        f"Hi there! Thanks for reaching out to {shop.get('name')}.\n\n"
        "You can:\n"
        "• Text photos of the damage for a fast AI-powered estimate, or\n"
        "• Reply with BOOK to schedule a FREE in-person inspection."
    )


# ============================================================
# FastAPI App + Twilio Webhook
# ============================================================

app = FastAPI()


@app.get("/")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "shops_loaded": len(SHOPS_BY_TOKEN),
        "has_openai": bool(client),
    }


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio SMS/MMS webhook.
    Expects a query parameter ?token=shop_xxx to identify the shop.
    """
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token)

    resp = MessagingResponse()

    if not shop:
        logger.error("Invalid or missing shop token: %s", token)
        resp.message(
            "This number is not configured correctly. Please contact the shop directly."
        )
        return Response(str(resp), media_type="application/xml")

    try:
        form = await request.form()
    except Exception:
        logger.exception("Failed to parse incoming Twilio form data.")
        resp.message(
            "Sorry, we couldn't read your message. Please try again or contact the shop directly."
        )
        return Response(str(resp), media_type="application/xml")

    from_phone = form.get("From", "")
    body = form.get("Body", "") or ""
    num_media_str = form.get("NumMedia", "0") or "0"

    try:
        num_media = int(num_media_str)
    except ValueError:
        num_media = 0

    logger.info(
        "Incoming SMS for shop=%s from=%s body=%r images=%d",
        shop.get("id"),
        from_phone,
        body,
        num_media,
    )

    try:
        if num_media > 0:
            # Collect all media URLs Twilio sent (MediaUrl0..MediaUrlN)
            media_urls: List[str] = []
            for i in range(num_media):
                url = form.get(f"MediaUrl{i}")
                if url:
                    media_urls.append(url)

            if not media_urls:
                logger.warning("NumMedia > 0 but no MediaUrl fields found.")
                resp.message(
                    "We couldn't read your photos. Please try sending them again, or reply with BOOK to schedule an in-person inspection."
                )
            else:
                estimate_text = await run_damage_estimator(shop, media_urls)
                full_message = (
                    f"AI Damage Estimate for {shop.get('name')}:\n\n{estimate_text}"
                )
                resp.message(full_message)
        else:
            reply_text = handle_text_message(shop, from_phone, body)
            resp.message(reply_text)
    except Exception:
        logger.exception("Unhandled error in sms_webhook.")
        resp.message(
            "Sorry, something went wrong while processing your message.\n\n"
            "Please try again in a few minutes or contact the shop directly."
        )

    return Response(str(resp), media_type="application/xml")
