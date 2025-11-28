import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# Google Calendar imports (optional but used when credentials are configured)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except ImportError:
    service_account = None
    build = None

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autobody-ai")

# ============================================================
# FastAPI + OpenAI setup
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

# OpenAI client (v1.x)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Shop + pricing config via SHOPS_JSON
# ============================================================

class LaborRates(BaseModel):
    body: float
    paint: float

class BaseFloor(BaseModel):
    minor_min: float
    minor_max: float
    moderate_min: float
    moderate_max: float
    severe_min: float
    severe_max: float

class PricingConfig(BaseModel):
    labor_rates: LaborRates
    materials_rate: float
    base_floor: BaseFloor

class ShopHours(BaseModel):
    monday: List[str]
    tuesday: List[str]
    wednesday: List[str]
    thursday: List[str]
    friday: List[str]
    saturday: List[str]
    sunday: List[str]

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str  # used as ?token=... in Twilio URL
    calendar_id: Optional[str] = None
    pricing: Optional[PricingConfig] = None
    hours: Optional[ShopHours] = None

def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError(
            "SHOPS_JSON env var is required. Example:\n"
            '[{"id":"miss","name":"Mississauga Collision Centre",'
            '"webhook_token":"shop_miss_123","calendar_id":"you@gmail.com",'
            '"pricing":{"labor_rates":{"body":95,"paint":105},'
            '"materials_rate":38,'
            '"base_floor":{"minor_min":350,"minor_max":650,'
            '"moderate_min":900,"moderate_max":1600,'
            '"severe_min":2000,"severe_max":5000}},'
            '"hours":{"monday":["9am-5pm"],"tuesday":["9am-5pm"],'
            '"wednesday":["9am-7pm"],"thursday":["9am-7pm"],'
            '"friday":["9am-7pm"],"saturday":["9am-5pm"],"sunday":["closed"]}}]'
        )
    try:
        data = json.loads(raw)
        shops_by_token: Dict[str, Shop] = {}
        for item in data:
            shop = Shop(**item)
            shops_by_token[shop.webhook_token] = shop
        logger.info("Loaded %d shops from SHOPS_JSON", len(shops_by_token))
        return shops_by_token
    except Exception as exc:
        logger.exception("Failed to parse SHOPS_JSON")
        raise RuntimeError(f"Invalid SHOPS_JSON: {exc}") from exc

SHOPS_BY_TOKEN = load_shops()

def get_shop_by_token(token: str) -> Shop:
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404, detail="Unknown shop token")
    return shop

# ============================================================
# Google Calendar helper
# ============================================================

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_calendar_service_cache: Optional[Any] = None

def get_calendar_service():
    global _calendar_service_cache
    if _calendar_service_cache is not None:
        return _calendar_service_cache
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON not set; calendar features disabled")
        return None
    if service_account is None or build is None:
        logger.warning("google-api-python-client not installed; calendar features disabled")
        return None
    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=GCAL_SCOPES
        )
        _calendar_service_cache = build("calendar", "v3", credentials=creds)
        return _calendar_service_cache
    except Exception as exc:
        logger.exception("Failed to create Google Calendar service: %s", exc)
        return None

def create_calendar_event(
    shop: Shop,
    customer_phone: str,
    customer_name: Optional[str],
    start_time: datetime,
    end_time: datetime,
    notes: str,
) -> Optional[str]:
    """
    Create a Google Calendar event for the given shop.
    Returns the HTML link to the event if successful, otherwise None.
    """
    if not shop.calendar_id:
        logger.warning("Shop %s has no calendar_id configured", shop.id)
        return None

    service = get_calendar_service()
    if not service:
        return None

    summary = f"AI estimate visit - {customer_phone}"
    if customer_name:
        summary = f"AI estimate visit - {customer_name} ({customer_phone})"

    event_body = {
        "summary": summary,
        "description": notes,
        "start": {"dateTime": start_time.isoformat()},
        "end": {"dateTime": end_time.isoformat()},
    }

    try:
        event = service.events().insert(calendarId=shop.calendar_id, body=event_body).execute()
        logger.info("Created calendar event %s for shop %s", event.get("id"), shop.id)
        return event.get("htmlLink")
    except Exception as exc:
        logger.exception("Failed to create calendar event: %s", exc)
        return None

# Simple in-memory state for booking flows (per phone number)
PENDING_BOOKINGS: Dict[str, Dict[str, Any]] = {}

# ============================================================
# AI helpers
# ============================================================

def build_pricing_hint(shop: Shop) -> str:
    if not shop.pricing:
        return "The shop has not provided detailed pricing, so give realistic ballpark ranges for Ontario body work."
    p = shop.pricing
    return (
        "Shop pricing model (CAD):\n"
        f"- Labor rate (body): ${p.labor_rates.body:.0f}/hr\n"
        f"- Labor rate (paint): ${p.labor_rates.paint:.0f}/hr\n"
        f"- Materials: ${p.materials_rate:.0f}/hr equivalent\n"
        f"- Typical repair ranges (before HST):\n"
        f"  * Minor: ${p.base_floor.minor_min:.0f} – ${p.base_floor.minor_max:.0f}\n"
        f"  * Moderate: ${p.base_floor.moderate_min:.0f} – ${p.base_floor.moderate_max:.0f}\n"
        f"  * Severe: ${p.base_floor.severe_min:.0f} – ${p.base_floor.severe_max:.0f}\n"
    )

def call_openai_chat(messages: List[Dict[str, Any]], temperature: float = 0.4) -> str:
    """
    Wrapper around OpenAI chat.completions.create with safe error handling.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("OpenAI chat error: %s", exc)
        # Fallback generic message
        return (
            "Our AI helper is temporarily unavailable. "
            "Please call the shop directly or try again in a few minutes."
        )

def build_damage_estimate(
    shop: Shop,
    image_url: str,
    user_message: Optional[str],
    customer_phone: str,
) -> str:
    """
    Hybrid smart mode for photo-based AI estimate.
    Uses GPT-4o with image input plus pricing hints.
    """
    pricing_hint = build_pricing_hint(shop)
    extra_text = user_message or ""

    system_prompt = (
        "You are an auto body damage estimator working for a collision centre in Ontario, Canada. "
        "You ONLY provide a preliminary estimate and severity rating based on the photo and text description. "
        "You must:\n"
        "1) Describe the visible damage in clear bullet points.\n"
        "2) Classify severity as Minor / Moderate / Severe.\n"
        "3) Give a realistic estimated price range in CAD, using the shop's pricing model.\n"
        "4) Clearly state that this is a visual, preliminary estimate only, NOT a final bill.\n"
        "5) Invite the customer to book a free in-person inspection.\n\n"
        f"{pricing_hint}\n"
        "Always keep the answer under ~220 words so it fits in a few SMS messages."
    )

    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Here is the customer's photo for damage estimation. "
                f"The customer's phone number is {customer_phone}. "
                "Customer's own description (if any) is appended below."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ]

    if extra_text:
        user_content.append(
            {
                "type": "text",
                "text": f"Customer's text description: {extra_text}",
            }
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        ai_text = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("OpenAI vision error: %s", exc)
        ai_text = (
            "Thanks for sending the photo. Our AI estimator is temporarily down, "
            "but our team can still help you. Please reply with a brief description of the damage "
            "or call the shop directly to book a free estimate."
        )

    # Save brief summary for potential booking flow
    short_summary = "AI estimate for collision damage based on customer photo."
    PENDING_BOOKINGS[customer_phone] = {
        "shop_id": shop.id,
        "summary": short_summary,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Add booking instructions
    ai_text += (
        "\n\nIf you’d like to book a FREE in-person inspection, "
        "reply with the word: BOOK"
    )
    return ai_text

def build_text_only_reply(shop: Shop, user_message: str, customer_phone: str) -> str:
    """
    Hybrid smart mode for text-only messages.
    Acts as a smart receptionist: answers questions and nudges for photos/booking.
    """
    system_prompt = (
        f"You are the SMS receptionist for {shop.name}, an auto body & collision centre in Ontario, Canada.\n"
        "Your goals:\n"
        "1) Be friendly, concise, and professional.\n"
        "2) Answer questions about estimates, repair timelines, and insurance in plain language.\n"
        "3) If they ask for a price or quote but haven’t sent photos, ask them to text clear photos of ALL damage.\n"
        "4) Encourage them to book a free in-person inspection.\n"
        "5) Keep replies under 3 SMS messages (~450 characters)."
    )
    user_prompt = (
        f"Customer phone: {customer_phone}\n"
        f"Customer message: {user_message}\n\n"
        "Reply in SMS style (short paragraphs, no markdown)."
    )

    reply = call_openai_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    # Light-touch instruction for booking
    if "BOOK" not in reply.upper():
        reply += (
            "\n\nTo book a FREE in-person inspection, "
            "you can also reply with the word: BOOK"
        )
    return reply

# ============================================================
# Twilio SMS Webhook (core entrypoint)
# ============================================================

@app.post("/sms-webhook", response_class=PlainTextResponse)
async def sms_webhook(request: Request, token: str = Query(...)) -> str:
    """
    Twilio will POST here for each incoming SMS/MMS.
    We respond with TwiML so Twilio can text the customer back.
    """
    shop = get_shop_by_token(token)

    form = await request.form()
    from_number: str = form.get("From", "Unknown")
    body: str = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", "0") or "0")
    image_url: Optional[str] = None
    if num_media > 0:
        image_url = form.get("MediaUrl0")

    logger.info(
        "Incoming SMS for shop %s from %s: body=%r, num_media=%s",
        shop.id,
        from_number,
        body,
        num_media,
    )

    twiml = MessagingResponse()

    # 1) Booking flow: user replies with BOOK / YES / NO
    upper_body = body.upper()
    if upper_body == "BOOK":
        # Offer a simple 30-minute slot tomorrow at 10am local (demo-friendly & deterministic)
        # In a real system you would query Google Calendar for real availability.
        now = datetime.utcnow()
        start = now + timedelta(days=1)
        # Approx: 10am Toronto ≈ 15:00 UTC (this is fine for a demo)
        start = start.replace(hour=15, minute=0, second=0, microsecond=0)
        end = start + timedelta(minutes=30)

        PENDING_BOOKINGS[from_number] = {
            "shop_id": shop.id,
            "summary": PENDING_BOOKINGS.get(from_number, {}).get(
                "summary", "AI estimate visit from SMS lead."
            ),
            "proposed_start": start.isoformat(),
            "proposed_end": end.isoformat(),
        }

        msg = (
            f"Great! We can book you for a FREE inspection at {shop.name}.\n"
            f"Proposed time: {start.strftime('%A %b %d, %I:%M %p')}.\n"
            "Reply YES to confirm this time or NO to pick a different time by phone."
        )
        twiml.message(msg)
        return str(twiml)

    if upper_body == "YES" and from_number in PENDING_BOOKINGS:
        pending = PENDING_BOOKINGS[from_number]
        start = datetime.fromisoformat(pending["proposed_start"])
        end = datetime.fromisoformat(pending["proposed_end"])
        summary = pending.get("summary", "AI estimate visit from SMS lead.")
        notes = (
            f"Customer phone: {from_number}\n"
            f"Summary: {summary}\n"
            f"Booked via AI SMS assistant on {datetime.utcnow().isoformat()} UTC."
        )

        event_link = create_calendar_event(
            shop=shop,
            customer_phone=from_number,
            customer_name=None,
            start_time=start,
            end_time=end,
            notes=notes,
        )

        if event_link:
            msg = (
                "You’re booked! We’ve reserved this time for your inspection. "
                "If you need to change it, please call the shop directly.\n\n"
                "Calendar ref: " + event_link
            )
        else:
            msg = (
                "You’re booked! We’ve reserved this time internally. "
                "If you need to change it, please call the shop directly."
            )

        twiml.message(msg)
        # Clear pending booking
        PENDING_BOOKINGS.pop(from_number, None)
        return str(twiml)

    if upper_body == "NO" and from_number in PENDING_BOOKINGS:
        PENDING_BOOKINGS.pop(from_number, None)
        msg = (
            f"No problem. Please call {shop.name} directly and we’ll arrange a time "
            "that works best for you."
        )
        twiml.message(msg)
        return str(twiml)

    # 2) Hybrid smart mode:
    #    - If they sent a photo, run full AI damage estimator.
    #    - If text only, use receptionist mode.
    if image_url:
        reply_text = build_damage_estimate(
            shop=shop,
            image_url=image_url,
            user_message=body,
            customer_phone=from_number,
        )
    else:
        reply_text = build_text_only_reply(
            shop=shop,
            user_message=body or "Customer sent a blank message.",
            customer_phone=from_number,
        )

    twiml.message(reply_text)
    return str(twiml)

# ============================================================
# Simple admin + health endpoints
# ============================================================

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

class ShopPublic(BaseModel):
    id: str
    name: str
    calendar_id: Optional[str] = None

@app.get("/admin/shops", response_model=List[ShopPublic])
def list_shops():
    return [
        ShopPublic(id=s.id, name=s.name, calendar_id=s.calendar_id)
        for s in SHOPS_BY_TOKEN.values()
                 ]
