import os
import json
import uuid
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# BASIC SETUP
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto-estimator")

app = FastAPI()

DEFAULT_TIMEZONE = ZoneInfo("America/Toronto")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if not GOOGLE_SERVICE_ACCOUNT_JSON:
    logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON not set – calendar bookings will fail.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. On Railway, attach Postgres and set DATABASE_URL."
    )

# ============================================================
# DATABASE
# ============================================================

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_key=True, index=True)
    shop_id = Column(String, index=True)
    from_number = Column(String, index=True)
    to_number = Column(String)
    severity = Column(String)
    estimate_min = Column(Float)
    estimate_max = Column(Float)
    currency = Column(String, default="CAD")
    areas = Column(Text)  # JSON list as string
    damage_types = Column(Text)  # JSON list as string
    summary = Column(Text)
    raw_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class BookingRequest(Base):
    __tablename__ = "booking_requests"

    id = Column(String, primary_key=True, index=True)
    shop_id = Column(String, index=True)
    estimate_id = Column(String, index=True)
    calendar_event_id = Column(String, nullable=True)
    customer_name = Column(String)
    customer_phone = Column(String)
    customer_email = Column(String)
    vehicle_details = Column(Text)
    preferred_datetime = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ============================================================
# SHOP CONFIG (SHOPS_JSON)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: Dict[str, Any]
    hours: Dict[str, List[str]]


def load_shops() -> Dict[str, ShopConfig]:
    """
    SHOPS_JSON example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123",
        "calendar_id": "shiran.bookings@gmail.com",
        "pricing": {
          "labor_rates": {
            "body": 95,
            "paint": 105
          },
          "materials_rate": 38,
          "base_floor": {
            "minor_min": 350,
            "minor_max": 650,
            "moderate_min": 900,
            "moderate_max": 1600,
            "severe_min": 2000,
            "severe_max": 5000
          }
        },
        "hours": {
          "monday": ["9am-5pm"],
          "tuesday": ["9am-5pm"],
          "wednesday": ["9am-7pm"],
          "thursday": ["9am-7pm"],
          "friday": ["9am-7pm"],
          "saturday": ["9am-5pm"],
          "sunday": ["closed"]
        }
      }
    ]
    """
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON env var is required")

    try:
        data = json.loads(raw)
        shops: Dict[str, ShopConfig] = {}
        for item in data:
            shop = ShopConfig(**item)
            shops[shop.webhook_token] = shop
        if not shops:
            raise RuntimeError("SHOPS_JSON parsed but no shops found")
        return shops
    except Exception as e:
        logger.exception("Failed to parse SHOPS_JSON")
        raise RuntimeError(f"Invalid SHOPS_JSON: {e}")


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()


# ============================================================
# GOOGLE CALENDAR
# ============================================================

_calendar_service = None


def get_calendar_service():
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def create_calendar_event(
    shop: ShopConfig,
    customer_name: str,
    customer_phone: str,
    customer_email: str,
    vehicle_details: str,
    start_dt: datetime,
) -> str:
    """
    Create a 60-minute booking event in the shop's Google Calendar.
    """
    service = get_calendar_service()

    end_dt = start_dt + timedelta(hours=1)
    event_body = {
        "summary": f"AI Estimate Booking - {customer_name}",
        "description": (
            f"Customer: {customer_name}\n"
            f"Phone: {customer_phone}\n"
            f"Email: {customer_email}\n"
            f"Vehicle: {vehicle_details}\n"
            f"Source: AI Damage Estimator"
        ),
        "start": {
            "dateTime": start_dt.astimezone(DEFAULT_TIMEZONE).isoformat(),
            "timeZone": str(DEFAULT_TIMEZONE),
        },
        "end": {
            "dateTime": end_dt.astimezone(DEFAULT_TIMEZONE).isoformat(),
            "timeZone": str(DEFAULT_TIMEZONE),
        },
    }

    event = (
        service.events()
        .insert(calendarId=shop.calendar_id, body=event_body, sendUpdates="all")
        .execute()
    )
    return event.get("id", "")


# ============================================================
# OPENAI VISION DAMAGE ESTIMATOR
# ============================================================

def run_damage_estimator(
    shop: ShopConfig,
    image_urls: List[str],
    user_text: str,
    from_number: str,
) -> Dict[str, Any]:
    """
    Send all received images + user text to OpenAI Vision and return a structured estimate.
    """

    pricing = shop.pricing or {}
    labor_body = pricing.get("labor_rates", {}).get("body")
    labor_paint = pricing.get("labor_rates", {}).get("paint")
    materials_rate = pricing.get("materials_rate")
    base_floor = pricing.get("base_floor", {})

    pricing_context = {
        "labor_body": labor_body,
        "labor_paint": labor_paint,
        "materials_rate": materials_rate,
        "base_floor": base_floor,
        "currency": "CAD",
        "region": "Ontario, Canada, year 2025",
    }

    system_prompt = (
        "You are an expert collision estimator for an Ontario (Canada) auto body shop in 2025. "
        "You look at vehicle damage photos and a brief text description and produce a realistic "
        "preliminary repair estimate.\n\n"
        "Always respond in **strict JSON** with this schema:\n"
        "{\n"
        '  "severity": "minor" | "moderate" | "severe",\n'
        '  "estimated_min": number,   // minimum cost in CAD\n'
        '  "estimated_max": number,   // maximum cost in CAD\n'
        '  "currency": "CAD",\n'
        '  "areas": [string],         // e.g. ["front bumper", "hood"]\n'
        '  "damage_types": [string],  // e.g. ["scratches", "deep dents"]\n'
        '  "summary": string,         // human-friendly summary\n'
        '  "recommendation": string   // next steps for the customer\n'
        "}\n\n"
        "Use the shop-specific pricing context provided. Stay honest and conservative. "
        "Do NOT mention that you are an AI. Do NOT include any text outside of the JSON."
    )

    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Here is the customer information and context:\n"
                    f"Phone number: {from_number}\n"
                    f"Shop name: {shop.name}\n\n"
                    "Customer description of damage (if any):\n"
                    f"{user_text or '(no description provided)'}\n\n"
                    "Shop pricing context (CAD, Ontario 2025):\n"
                    + json.dumps(pricing_context, indent=2)
                ),
            },
        ]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            }
            for url in image_urls
        ],
    }

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            user_prompt,
        ],
        temperature=0.1,
    )

    raw_content = completion.choices[0].message.content.strip()
    logger.info(f"Raw OpenAI response: {raw_content[:300]}")

    # Make sure we only parse JSON (strip any accidental text)
    json_str = raw_content
    if "{" in raw_content and "}" in raw_content:
        json_str = raw_content[raw_content.index("{") : raw_content.rindex("}") + 1]

    try:
        data = json.loads(json_str)
    except Exception:
        # Fallback if parsing fails
        data = {
            "severity": "moderate",
            "estimated_min": pricing_context["base_floor"].get("moderate_min", 900),
            "estimated_max": pricing_context["base_floor"].get("moderate_max", 1600),
            "currency": "CAD",
            "areas": [],
            "damage_types": [],
            "summary": "Preliminary estimate based on visible damage.",
            "recommendation": "Visit the shop for a full teardown and detailed estimate.",
        }

    # Ensure required fields
    data.setdefault("currency", "CAD")
    data.setdefault("areas", [])
    data.setdefault("damage_types", [])
    data.setdefault("summary", "Preliminary estimate based on photos.")
    data.setdefault(
        "recommendation",
        "We recommend booking an in-person inspection to confirm this estimate.",
    )

    return {"parsed": data, "raw": raw_content}


# ============================================================
# SMS CONVERSATION STATE (IN-MEMORY)
# ============================================================

class ConversationState(BaseModel):
    last_stage: str = "start"  # "start" | "estimated" | "awaiting_booking" | "booked"
    last_estimate_id: Optional[str] = None


CONVERSATIONS: Dict[str, ConversationState] = {}


def get_conv_state(phone: str) -> ConversationState:
    if phone not in CONVERSATIONS:
        CONVERSATIONS[phone] = ConversationState()
    return CONVERSATIONS[phone]


# ============================================================
# HELPERS
# ============================================================

def get_image_urls_from_form(form) -> List[str]:
    """
    Collect all Twilio MediaUrlX fields as a list of URLs.
    """
    num_media_str = form.get("NumMedia") or "0"
    try:
        num_media = int(num_media_str)
    except ValueError:
        num_media = 0

    urls: List[str] = []
    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if url:
            urls.append(url)
    return urls


BOOKING_REGEX = re.compile(
    r"^book\s+(?P<datetime>.+?)\s*;\s*(?P<name>.+?)\s*;\s*(?P<email>.+?)\s*;\s*(?P<vehicle>.+)$",
    re.IGNORECASE,
)


def parse_booking_message(body: str) -> Optional[Dict[str, str]]:
    """
    Expected format from customer:

    book 2025-11-30 2:30pm; John Doe; john@example.com; 2018 Honda Civic grey

    Returns dict or None if not matched.
    """
    match = BOOKING_REGEX.match(body.strip())
    if not match:
        return None
    return match.groupdict()


def parse_datetime_in_toronto(raw: str) -> Optional[datetime]:
    """
    Very forgiving datetime parser for customer text.
    Examples:
      "2025-11-30 2:30pm"
      "Nov 30 3pm"
      "tomorrow 10am"  (won't handle all natural language, but we try a few formats)
    """
    raw = raw.strip()

    # Simple cases first
    for fmt in [
        "%Y-%m-%d %I:%M%p",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I%p",
        "%Y-%m-%d",
        "%b %d %I:%M%p",
        "%b %d %I%p",
        "%b %d",
    ]:
        try:
            dt = datetime.strptime(raw, fmt)
            # If no year, assume this year
            if "%Y" not in fmt:
                dt = dt.replace(year=datetime.now(DEFAULT_TIMEZONE).year)
            return dt.replace(tzinfo=DEFAULT_TIMEZONE)
        except ValueError:
            continue

    return None


def shop_hours_to_text(shop: ShopConfig) -> str:
    """Convert hours dict to a human-readable summary."""
    lines = []
    for day in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]:
        slots = shop.hours.get(day, [])
        label = day.capitalize()
        if not slots or "closed" in [s.lower() for s in slots]:
            lines.append(f"{label}: Closed")
        else:
            lines.append(f"{label}: {', '.join(slots)}")
    return "\n".join(lines)


# ============================================================
# ROOT HEALTHCHECK
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Damage Estimator is running"}


# ============================================================
# TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Twilio webhook endpoint.
    URL in Twilio MUST be:  https://.../sms-webhook?token=shop_miss_123
    """
    # 1) Identify shop by token
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404, detail="Unknown shop token")

    # 2) Parse Twilio form data
    form = await request.form()
    from_number = form.get("From") or ""
    to_number = form.get("To") or ""
    body = (form.get("Body") or "").strip()
    image_urls = get_image_urls_from_form(form)

    logger.info(
        f"Incoming SMS: from={from_number}, to={to_number}, body={body!r}, images={len(image_urls)}"
    )

    resp = MessagingResponse()
    session = SessionLocal()

    try:
        conv = get_conv_state(from_number)

        # CASE 1: Customer sent photos -> run AI estimator
        if image_urls:
            try:
                ai_result = run_damage_estimator(
                    shop=shop,
                    image_urls=image_urls,
                    user_text=body,
                    from_number=from_number,
                )
                parsed = ai_result["parsed"]
                raw = ai_result["raw"]
            except Exception as e:
                logger.exception("Error during OpenAI damage estimation")
                msg = resp.message(
                    "Sorry, something went wrong while analyzing the pictures. "
                    "Please try again in a few minutes or text us directly."
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            estimate_id = str(uuid.uuid4())
            est = DamageEstimate(
                id=estimate_id,
                shop_id=shop.id,
                from_number=from_number,
                to_number=to_number,
                severity=parsed.get("severity"),
                estimate_min=parsed.get("estimated_min"),
                estimate_max=parsed.get("estimated_max"),
                currency=parsed.get("currency", "CAD"),
                areas=json.dumps(parsed.get("areas", [])),
                damage_types=json.dumps(parsed.get("damage_types", [])),
                summary=parsed.get("summary"),
                raw_response=raw,
            )
            session.add(est)
            session.commit()

            conv.last_stage = "estimated"
            conv.last_estimate_id = estimate_id

            hours_text = shop_hours_to_text(shop)

            pretty_min = int(parsed["estimated_min"])
            pretty_max = int(parsed["estimated_max"])

            reply_text = (
                f"AI Damage Estimate for {shop.name}\n\n"
                f"Severity: {parsed['severity'].capitalize()}\n"
                f"Estimated Cost ({parsed['currency']}): ${pretty_min:,} – ${pretty_max:,}\n\n"
                f"{parsed['summary']}\n\n"
                "Next step: Book a free in-person inspection so the team can confirm this estimate.\n\n"
                "To book, reply in this format:\n"
                "book 2025-11-30 2:30pm; Your Name; you@example.com; Year Make Model colour\n\n"
                "Shop hours:\n"
                f"{hours_text}"
            )
            resp.message(reply_text)
            return PlainTextResponse(str(resp), media_type="application/xml")

        # CASE 2: Customer is trying to book (no images)
        if body.lower().startswith("book"):
            booking_data = parse_booking_message(body)
            if not booking_data:
                msg = resp.message(
                    "Booking format not recognized.\n\n"
                    "Please reply like this:\n"
                    "book 2025-11-30 2:30pm; Your Name; you@example.com; 2018 Honda Civic grey"
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            if not conv.last_estimate_id:
                msg = resp.message(
                    "Before booking, please send 1–3 clear photos of the damage so we can "
                    "generate an AI estimate for you."
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            dt_raw = booking_data["datetime"]
            dt = parse_datetime_in_toronto(dt_raw)
            if not dt:
                msg = resp.message(
                    "I couldn't understand that date/time.\n"
                    "Please use a format like: 2025-11-30 2:30pm"
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            customer_name = booking_data["name"]
            customer_email = booking_data["email"]
            vehicle = booking_data["vehicle"]

            # Create calendar event
            try:
                event_id = create_calendar_event(
                    shop=shop,
                    customer_name=customer_name,
                    customer_phone=from_number,
                    customer_email=customer_email,
                    vehicle_details=vehicle,
                    start_dt=dt,
                )
            except Exception as e:
                logger.exception("Error creating Google Calendar event")
                event_id = ""

            booking_id = str(uuid.uuid4())
            booking = BookingRequest(
                id=booking_id,
                shop_id=shop.id,
                estimate_id=conv.last_estimate_id,
                calendar_event_id=event_id,
                customer_name=customer_name,
                customer_phone=from_number,
                customer_email=customer_email,
                vehicle_details=vehicle,
                preferred_datetime=dt,
            )
            session.add(booking)
            session.commit()

            conv.last_stage = "booked"

            reply = (
                f"Thanks {customer_name}! Your visit request has been received.\n\n"
                f"Requested time: {dt.astimezone(DEFAULT_TIMEZONE).strftime('%a %b %d, %I:%M %p')}\n"
                "The shop will confirm or contact you if an adjustment is needed.\n\n"
                f"{shop.name}\n"
                "If you need to change anything, just reply to this message."
            )
            resp.message(reply)
            return PlainTextResponse(str(resp), media_type="application/xml")

        # CASE 3: No images + not 'book' -> onboarding / help
        if conv.last_stage == "start":
            msg = resp.message(
                f"Thanks for contacting {shop.name}! 👋\n\n"
                "To get a free AI-powered estimate, please reply with 1–3 clear photos of the damage.\n\n"
                "Optional: You can also describe what happened (e.g. 'rear-ended at low speed')."
            )
            return PlainTextResponse(str(resp), media_type="application/xml")

        if conv.last_stage in ("estimated", "awaiting_booking"):
            msg = resp.message(
                "We already sent your AI estimate.\n\n"
                "To book a visit, reply in this format:\n"
                "book 2025-11-30 2:30pm; Your Name; you@example.com; Year Make Model colour"
            )
            return PlainTextResponse(str(resp), media_type="application/xml")

        if conv.last_stage == "booked":
            msg = resp.message(
                "We already have your booking request on file. "
                "If you need to change something, reply with the details and the shop will follow up."
            )
            return PlainTextResponse(str(resp), media_type="application/xml")

        # Fallback (should not hit)
        resp.message(
            "Thanks for your message. Please send 1–3 photos of the damage to get started."
        )
        return PlainTextResponse(str(resp), media_type="application/xml")

    finally:
        session.close()
