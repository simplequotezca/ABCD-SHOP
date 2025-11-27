import os
import json
import uuid
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo
import requests

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Environment Variables
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON is not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Local timezone for human-readable times
LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI Client (SAFE INIT)
# ============================================================

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing — AI estimator will be disabled.")
    openai_client: Optional[OpenAI] = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Database Models
# ============================================================


class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    shop_id = Column(String, index=True)
    phone_number = Column(String, index=True)
    user_message = Column(Text)

    vin = Column(String, nullable=True)
    vin_decoded = Column(Text, nullable=True)

    severity = Column(String)
    estimated_cost_min = Column(Float)
    estimated_cost_max = Column(Float)

    vehicle_info = Column(Text)
    ai_raw_json = Column(Text)
    ai_summary_text = Column(Text)


class BookingRequest(Base):
    __tablename__ = "booking_requests"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    shop_id = Column(String, index=True)
    phone_number = Column(String, index=True)
    estimate_id = Column(String, nullable=True)

    step = Column(String, index=True)  # 'choice', 'awaiting_datetime', 'completed'
    preferred_text = Column(Text)      # raw text user sent ("tomorrow 3pm")
    calendar_event_id = Column(String) # ID from Google Calendar


Base.metadata.create_all(bind=engine)

# ============================================================
# Shop Loading
# ============================================================


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    hours: Optional[Dict[str, Any]] = None


def load_shops() -> Dict[str, Shop]:
    """
    SHOPS_JSON example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123",
        "calendar_id": "shiran.bookings@gmail.com",
        "pricing": {...},
        "hours": {...}
      }
    ]
    """
    raw = json.loads(SHOPS_JSON)
    shops: Dict[str, Shop] = {}

    for entry in raw:
        shop = Shop(**entry)
        shops[shop.webhook_token] = shop

    logger.info("Loaded %d shops from SHOPS_JSON", len(shops))
    return shops


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN Handling
# ============================================================

VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"


def extract_vin(text: str) -> Optional[str]:
    """
    Grab a 17-character VIN from the SMS body if present.
    We strip spaces and force uppercase first.
    """
    clean = text.replace(" ", "").upper()
    match = re.search(VIN_REGEX, clean)
    return match.group(1) if match else None


def decode_vin(vin: str) -> Dict[str, Any]:
    """
    Uses NHTSA public API to decode a VIN.
    Safe to fail – will just return minimal info.
    """
    try:
        url = (
            f"https://vpic.nhtsa.dot.gov/api/vehicles/"
            f"DecodeVinValuesExtended/{vin}?format=json"
        )
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        result = res.json()["Results"][0]

        return {
            "vin": vin,
            "year": result.get("ModelYear"),
            "make": result.get("Make"),
            "model": result.get("Model"),
            "trim": result.get("Trim"),
            "body_class": result.get("BodyClass"),
            "engine": result.get("EngineCylinders"),
            "fuel_type": result.get("FuelTypePrimary"),
            "plant": result.get("PlantCity"),
            "series": result.get("Series"),
        }
    except Exception as e:
        logger.warning("VIN decode failed for %s: %s", vin, e)
        return {"vin": vin, "error": "decode_failed"}

# ============================================================
# Google Calendar Helpers
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]
_calendar_service = None


def get_calendar_service():
    """
    Lazy-init Google Calendar service from service account JSON.
    """
    global _calendar_service
    if _calendar_service:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES
    )

    _calendar_service = build(
        "calendar", "v3", credentials=creds, cache_discovery=False
    )
    return _calendar_service


def schedule_calendar_event(
    shop: Shop,
    phone_number: str,
    estimate_id: str,
    text: str,
):
    """
    Parse the user's date/time text, check shop's Google Calendar for conflicts,
    and create a 1-hour event if free.
    """
    if not shop.calendar_id:
        raise RuntimeError("Calendar not configured for this shop.")

    # Parse natural date/time like "tomorrow 3pm"
    try:
        dt = date_parser.parse(text, fuzzy=True)
    except Exception as e:
        logger.warning("Date parse failed for '%s': %s", text, e)
        raise ValueError("Invalid date/time format") from e

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    else:
        dt = dt.astimezone(LOCAL_TZ)

    start = dt
    end = start + timedelta(hours=1)

    service = get_calendar_service()

    # Check for conflicts
    events = (
        service.events()
        .list(
            calendarId=shop.calendar_id,
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
        )
        .execute()
        .get("items", [])
    )

    if events:
        logger.info("Calendar conflict for %s between %s and %s", shop.id, start, end)
        raise RuntimeError("Time slot unavailable")

    # Build a rich event description with all context
    description_lines = [
        f"SMS lead from {phone_number}",
        f"Estimate ID: {estimate_id}",
        "",
        "Customer preferred time text:",
        text,
    ]
    description = "\n".join(description_lines)

    body = {
        "summary": f"Estimate Appointment - {shop.name}",
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": str(LOCAL_TZ)},
        "end": {"dateTime": end.isoformat(), "timeZone": str(LOCAL_TZ)},
    }

    event = (
        service.events()
        .insert(calendarId=shop.calendar_id, body=body)
        .execute()
    )

    logger.info(
        "Created calendar event %s for shop %s at %s",
        event.get("id"),
        shop.id,
        start,
    )

    return start, end, event.get("id")

# ============================================================
# AI Estimator
# ============================================================


def build_prompt(shop: Shop, user_text: str, vin_data: dict) -> str:
    """
    Create a consistent prompt for the estimator model.
    Left/right is clearly defined to avoid confusion.
    """
    return f"""
You are a professional collision estimator working in Ontario in 2025.

CRITICAL LEFT/RIGHT RULE:
- "Left" always means the DRIVER side of the vehicle.
- "Right" always means the PASSENGER side of the vehicle.
- Never mix these up. If the image is unclear, say so in notes.

VEHICLE INFO (from VIN if available):
{json.dumps(vin_data or {}, indent=2)}

CUSTOMER TEXT DESCRIPTION:
{user_text}

SHOP PRICING (labor & materials):
{json.dumps(shop.pricing or {}, indent=2)}

Your task:
1. Look at the photo(s) and text.
2. Identify which panels/areas are damaged (e.g. front bumper, left fender, right quarter panel, trunk, hood, etc.).
3. Classify damage types (scrape, paint damage, light dent, deep dent, structural deformation, cracked bumper, broken lamp, etc.).
4. Use Ontario 2025 typical body shop pricing along with SHOP PRICING to estimate a realistic repair cost range.

Return ONLY valid JSON in this format (no extra text):

{{
  "severity": "minor | moderate | severe",
  "estimated_cost_min": 0,
  "estimated_cost_max": 0,
  "areas": ["front bumper", "hood"],
  "damage_types": ["deep dent", "scrape"],
  "notes": "Short bullet summary of key damage. Be honest about uncertainty.",
  "recommendation": "Short recommendation, e.g. safe to drive or tow recommended."
}}
"""


def call_ai_estimator(
    image_url: str,
    shop: Shop,
    text: str,
    vin_data: Optional[dict],
) -> Dict[str, Any]:
    """
    Call OpenAI Vision (gpt-4o) with image + text and parse JSON response.
    """
    if not openai_client:
        raise RuntimeError("OpenAI not configured")

    prompt = build_prompt(shop, text, vin_data or {})

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a collision estimator. Always reply with pure JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )
    except Exception as e:
        logger.exception("OpenAI call failed")
        raise RuntimeError(f"OpenAI error: {e}") from e

    raw = completion.choices[0].message.content or ""

    # Be defensive in case the model wraps JSON in extra text
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            logger.error("AI returned non-JSON content: %s", raw)
            raise RuntimeError("AI returned invalid JSON")
        try:
            return json.loads(raw[start : end + 1])
        except Exception as e:
            logger.error("Failed to parse trimmed JSON: %s", e)
            raise RuntimeError("AI JSON parse failed") from e

# ============================================================
# SMS Formatting Helpers
# ============================================================


def sms_estimate(shop: Shop, result: dict, est_id: str) -> str:
    """
    Build the SMS text that goes back to the customer with the estimate.
    """
    severity = result.get("severity", "unknown").title()
    cmin = result.get("estimated_cost_min") or 0
    cmax = result.get("estimated_cost_max") or 0
    areas = ", ".join(result.get("areas") or [])
    dmg = ", ".join(result.get("damage_types") or [])
    notes = result.get("notes") or ""
    recommendation = result.get("recommendation") or ""

    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {severity}\n"
        f"Estimated Cost (Ontario 2025): ${cmin:,.0f} – ${cmax:,.0f}\n"
        f"Areas: {areas or 'n/a'}\n"
        f"Damage Types: {dmg or 'n/a'}\n\n"
        f"Notes: {notes}\n"
        f"Recommendation: {recommendation}\n\n"
        f"This is a visual, preliminary estimate only. "
        f"Reply 1 to book an in-person assessment.\n\n"
        f"Estimate ID: {est_id}"
    )


def sms_confirmation(shop: Shop, dt: datetime) -> str:
    dt_local = dt.astimezone(LOCAL_TZ)
    return (
        f"You're booked at {shop.name} on "
        f"{dt_local.strftime('%A, %B %d at %I:%M %p')}.\n"
        f"We'll see you then!"
    )

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Auto Body AI Estimator Backend")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Auto Body AI Estimator running."}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio webhook endpoint.

    URL pattern (per shop):
    https://<railway-app-url>/sms-webhook?token=shop_miss_123
    """
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        logger.warning("Invalid or missing token: %s", token)
        raise HTTPException(status_code=403, detail="Invalid token")

    shop = SHOPS_BY_TOKEN[token]

    # Twilio form payload
    form = await request.form()
    from_number = form.get("From")
    raw_body = form.get("Body") or ""
    body = raw_body.strip()
    media_count = int(form.get("NumMedia") or "0")

    logger.info(
        "Incoming SMS from %s for shop %s (media_count=%s)",
        from_number,
        shop.id,
        media_count,
    )

    session = SessionLocal()
    tw = MessagingResponse()

    try:
        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        # ====================================================
        # 1) PHOTO RECEIVED – generate AI estimate
        # ====================================================
        if media_count > 0:
            image_url = form.get("MediaUrl0")
            logger.info("Received image from %s: %s", from_number, image_url)

            if not openai_client:
                logger.error("OpenAI client is not configured; cannot estimate.")
                tw.message(
                    "AI estimator is temporarily unavailable. "
                    "Please call the shop directly for an estimate."
                )
                return Response(str(tw), media_type="application/xml")

            try:
                result = call_ai_estimator(image_url, shop, raw_body, vin_data)
            except Exception as e:
                logger.exception("AI error while generating estimate")
                tw.message(
                    "Error generating estimate right now. "
                    "Please try again in a few minutes."
                )
                return Response(str(tw), media_type="application/xml")

            est_id = str(uuid.uuid4())

            estimate_record = DamageEstimate(
                id=est_id,
                shop_id=shop.id,
                phone_number=from_number,
                user_message=raw_body,
                vin=vin,
                vin_decoded=json.dumps(vin_data) if vin_data else None,
                severity=result.get("severity"),
                estimated_cost_min=result.get("estimated_cost_min") or 0.0,
                estimated_cost_max=result.get("estimated_cost_max") or 0.0,
                vehicle_info=raw_body,
                ai_raw_json=json.dumps(result),
                ai_summary_text=result.get("notes") or "",
            )
            session.add(estimate_record)
            session.commit()

            # Initialize booking flow state
            booking = BookingRequest(
                id=str(uuid.uuid4()),
                shop_id=shop.id,
                phone_number=from_number,
                estimate_id=est_id,
                step="choice",
                preferred_text="",
                calendar_event_id="",
            )
            session.add(booking)
            session.commit()

            tw.message(sms_estimate(shop, result, est_id))
            return Response(str(tw), media_type="application/xml")

        # ====================================================
        # 2) USER REPLIED "1" – start booking flow
        # ====================================================
        if body.lower() in {"1", "book", "one"}:
            booking = (
                session.query(BookingRequest)
                .filter(
                    BookingRequest.shop_id == shop.id,
                    BookingRequest.phone_number == from_number,
                )
                .order_by(BookingRequest.created_at.desc())
                .first()
            )

            if not booking:
                tw.message("Please send damage photos first so we can create an estimate.")
                return Response(str(tw), media_type="application/xml")

            booking.step = "awaiting_datetime"
            session.commit()

            tw.message(
                "Great! To book your visit, what day/time works best?\n"
                "Examples:\n"
                "- Tomorrow at 3pm\n"
                "- Friday 10am\n"
                "- Tuesday morning"
            )
            return Response(str(tw), media_type="application/xml")

        # ====================================================
        # 3) USER PROVIDING DATE/TIME – create calendar event
        # ====================================================
        booking = (
            session.query(BookingRequest)
            .filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == from_number,
                BookingRequest.step == "awaiting_datetime",
            )
            .order_by(BookingRequest.created_at.desc())
            .first()
        )

        if booking:
            booking.preferred_text = raw_body
            session.commit()

            try:
                start, end, event_id = schedule_calendar_event(
                    shop, from_number, booking.estimate_id, raw_body
                )
            except ValueError:
                tw.message(
                    "Sorry, I couldn’t understand that time. "
                    "Please try something like 'Tomorrow at 3pm' or 'Friday 10am'."
                )
                return Response(str(tw), media_type="application/xml")
            except Exception as e:
                logger.warning("Calendar scheduling error: %s", e)
                tw.message(
                    "That time isn't available or calendar is not configured. "
                    "Please suggest another time."
                )
                return Response(str(tw), media_type="application/xml")

            booking.step = "completed"
            booking.calendar_event_id = event_id
            session.commit()

            tw.message(sms_confirmation(shop, start))
            return Response(str(tw), media_type="application/xml")

        # ====================================================
        # 4) DEFAULT – instructions
        # ====================================================
        tw.message(
            "Welcome to our AI damage estimator.\n"
            "Please send 1–3 clear photos of the vehicle damage to begin."
        )
        return Response(str(tw), media_type="application/xml")

    finally:
        session.close()
