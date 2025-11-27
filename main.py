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
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo
import requests

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# ENV + DATABASE
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
LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI — v1 safe init
# ============================================================

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set – OpenAI calls will fail")
    openai_client: Optional[OpenAI] = None
else:
    # Explicit api_key, no extra kwargs (so we never hit the 'proxies' error)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# SQLAlchemy Models
# ============================================================

class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

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

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    shop_id = Column(String, index=True)
    phone_number = Column(String, index=True)
    estimate_id = Column(String, nullable=True)

    step = Column(String, index=True)  # "choice", "awaiting_datetime", "completed"
    preferred_text = Column(Text, nullable=True)
    calendar_event_id = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)

# ============================================================
# Shop Config
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    hours: Optional[Dict[str, Any]] = None


def load_shops() -> Dict[str, Shop]:
    raw = json.loads(SHOPS_JSON)
    shops: Dict[str, Shop] = {}
    for item in raw:
        shop = Shop(**item)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN DECODER
# ============================================================

# 17 chars, skipping I/O/Q as per VIN spec
VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"


def extract_vin(text: str) -> Optional[str]:
    """
    Detect VIN anywhere in the message.
    Customer can just paste the 17-digit VIN – no need for 'VIN:'.
    """
    compact = text.replace(" ", "").upper()
    match = re.search(VIN_REGEX, compact)
    return match.group(1) if match else None


def decode_vin(vin: str) -> Dict[str, Any]:
    """Decode VIN using free NHTSA API."""
    try:
        url = (
            "https://vpic.nhtsa.dot.gov/api/vehicles/"
            f"DecodeVinValuesExtended/{vin}?format=json"
        )
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()["Results"][0]

        return {
            "vin": vin,
            "year": data.get("ModelYear"),
            "make": data.get("Make"),
            "model": data.get("Model"),
            "trim": data.get("Trim"),
            "body_class": data.get("BodyClass"),
            "engine": data.get("EngineCylinders"),
            "fuel_type": data.get("FuelTypePrimary"),
            "plant": data.get("PlantCity"),
            "series": data.get("Series"),
        }
    except Exception as e:
        logger.warning(f"VIN decode failed for {vin}: {e}")
        return {"vin": vin, "error": "VIN decoding failed"}

# ============================================================
# Google Calendar Helpers
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]
_calendar_service = None


def get_calendar_service():
    global _calendar_service
    if _calendar_service:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

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
    estimate_id: Optional[str],
    requested_text: str,
):
    """Parse date/time + book 1-hour appointment if free."""
    if not shop.calendar_id:
        raise RuntimeError("Shop does not have a calendar_id configured")

    try:
        parsed_dt = date_parser.parse(requested_text, fuzzy=True)
    except Exception:
        raise ValueError("Invalid date/time")

    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=LOCAL_TZ)

    start_dt = parsed_dt
    end_dt = start_dt + timedelta(hours=1)

    service = get_calendar_service()
    events = (
        service.events()
        .list(
            calendarId=shop.calendar_id,
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
        )
        .execute()
        .get("items", [])
    )

    if events:
        raise RuntimeError("Requested time is not available")

    event_body = {
        "summary": "Auto Body Estimate Appointment",
        "description": f"SMS lead from {phone_number}. Estimate ID: {estimate_id}",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": str(LOCAL_TZ)},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": str(LOCAL_TZ)},
    }

    event = (
        service.events()
        .insert(calendarId=shop.calendar_id, body=event_body)
        .execute()
    )
    return start_dt, end_dt, event.get("id")

# ============================================================
# AI DAMAGE ESTIMATOR
# ============================================================

def build_ai_prompt(
    shop: Shop,
    user_text: str,
    vehicle_hint: str,
    pricing_hint: Optional[Dict[str, Any]],
    vin_data: Optional[Dict[str, Any]],
) -> str:
    pricing_str = json.dumps(pricing_hint, indent=2) if pricing_hint else "null"
    vin_str = json.dumps(vin_data, indent=2) if vin_data else "{}"

    return f"""
You are an expert collision estimator in Ontario, Canada (2025).

CRITICAL LEFT/RIGHT RULE:
- Always interpret left/right based on DRIVER'S point of view sitting in the driver seat facing forward.
- Driver side = left.
- Passenger side = right.

VEHICLE DETAILS (decoded from VIN if provided):
{vin_str}

USER TEXT:
{user_text}

SHOP:
{shop.name}

PRICING:
{pricing_str}

Return ONLY valid JSON using this schema:

{{
  "severity": "Minor" | "Moderate" | "Severe" | "Total Loss",
  "estimated_cost_min": number,
  "estimated_cost_max": number,
  "areas": [ "... list of areas ..." ],
  "damage_types": [ "... list of damage types ..." ],
  "notes": "short explanation",
  "recommendation": "short next-step advice"
}}
"""


def call_ai_estimator(
    image_url: str,
    shop: Shop,
    user_text: str,
    vehicle_hint: str,
    vin_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if openai_client is None:
        raise RuntimeError("OpenAI client is not configured (missing OPENAI_API_KEY)")

    prompt = build_ai_prompt(shop, user_text, vehicle_hint, shop.pricing, vin_data)

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0.2,
    )

    content = completion.choices[0].message.content

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("AI did not return JSON")
        return json.loads(content[start : end + 1])

# ============================================================
# Formatting Messages (polished & locked in)
# ============================================================

def format_estimate_sms(shop: Shop, estimate: Dict[str, Any], estimate_id: str) -> str:
    severity = estimate.get("severity") or "Unknown"
    min_cost = estimate.get("estimated_cost_min")
    max_cost = estimate.get("estimated_cost_max")

    try:
        cost_str = f"${min_cost:,.0f} – ${max_cost:,.0f}"
    except Exception:
        cost_str = "N/A"

    areas = ", ".join(estimate.get("areas") or [])
    damages = ", ".join(estimate.get("damage_types") or [])

    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {severity}\n"
        f"Estimated Cost (Ontario 2025): {cost_str}\n\n"
        f"Areas: {areas}\n"
        f"Damage Types: {damages}\n\n"
        "This is a visual, preliminary estimate only. "
        "Final pricing may vary after an in-person inspection.\n\n"
        "Reply 1 to schedule an appointment.\n\n"
        f"Estimate ID: {estimate_id}"
    )


def format_booking_confirmation(shop: Shop, start_dt: datetime) -> str:
    date = start_dt.strftime("%A, %B %d")
    time = start_dt.strftime("%I:%M %p").lstrip("0")
    return (
        f"You're booked at {shop.name} on {date} at {time}.\n\n"
        "If you need to make any changes, please contact the shop directly."
    )

# ============================================================
# FastAPI + Twilio Webhook
# ============================================================

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=403, detail="Invalid token")

    form = await request.form()
    from_number = form.get("From")
    raw_body = form.get("Body") or ""
    body = raw_body.strip()
    num_media = int(form.get("NumMedia") or "0")

    session = SessionLocal()
    twiml = MessagingResponse()

    try:
        # -------------------------------------------------------
        # VIN auto-detection (works on ANY text)
        # -------------------------------------------------------
        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        # -------------------------------------------------------
        # CASE 1: MMS with photos → generate AI estimate
        # -------------------------------------------------------
        if num_media > 0:
            image_url = form.get("MediaUrl0")
            if not image_url:
                twiml.message(
                    "We couldn't read the photo. Please try sending it again."
                )
                return Response(str(twiml), media_type="application/xml")

            try:
                estimate_json = call_ai_estimator(
                    image_url=image_url,
                    shop=shop,
                    user_text=raw_body,
                    vehicle_hint=raw_body,
                    vin_data=vin_data,
                )
            except Exception as e:
                logger.exception(f"AI estimator failed: {e}")
                twiml.message(
                    "We ran into an issue generating the estimate. "
                    "Please try again in a few minutes or call the shop."
                )
                return Response(str(twiml), media_type="application/xml")

            estimate_id = str(uuid.uuid4())

            # Save estimate
            try:
                session.add(
                    DamageEstimate(
                        id=estimate_id,
                        shop_id=shop.id,
                        phone_number=from_number,
                        user_message=raw_body,
                        vin=vin,
                        vin_decoded=json.dumps(vin_data) if vin_data else None,
                        severity=estimate_json.get("severity"),
                        estimated_cost_min=estimate_json.get("estimated_cost_min"),
                        estimated_cost_max=estimate_json.get("estimated_cost_max"),
                        vehicle_info=raw_body,
                        ai_raw_json=json.dumps(estimate_json),
                        ai_summary_text="",
                    )
                )
                session.commit()
            except Exception as e:
                logger.exception(f"Failed to save estimate: {e}")
                session.rollback()

            # Create booking request in "choice" step
            try:
                session.add(
                    BookingRequest(
                        id=str(uuid.uuid4()),
                        shop_id=shop.id,
                        phone_number=from_number,
                        estimate_id=estimate_id,
                        step="choice",
                    )
                )
                session.commit()
            except Exception as e:
                logger.exception(f"Failed to create BookingRequest: {e}")
                session.rollback()

            twiml.message(format_estimate_sms(shop, estimate_json, estimate_id))
            return Response(str(twiml), media_type="application/xml")

        # -------------------------------------------------------
        # CASE 2: user replies "1" → ask for date/time
        # -------------------------------------------------------
        if body.lower() in {"1", "one", "book"}:
            br = (
                session.query(BookingRequest)
                .filter(
                    BookingRequest.shop_id == shop.id,
                    BookingRequest.phone_number == from_number,
                )
                .order_by(BookingRequest.created_at.desc())
                .first()
            )

            if not br:
                twiml.message(
                    "Please send clear photos of the damage first to get an estimate."
                )
                return Response(str(twiml), media_type="application/xml")

            br.step = "awaiting_datetime"
            session.commit()

            twiml.message(
                "Great! What day and time works best for you?\n\n"
                "Examples:\n"
                '"Tuesday at 3pm"\n'
                '"Tomorrow at 10am"\n'
                '"Friday morning"'
            )
            return Response(str(twiml), media_type="application/xml")

        # -------------------------------------------------------
        # CASE 3: waiting for date/time → try to book
        # -------------------------------------------------------
        br_waiting = (
            session.query(BookingRequest)
            .filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == from_number,
                BookingRequest.step == "awaiting_datetime",
            )
            .order_by(BookingRequest.created_at.desc())
            .first()
        )

        if br_waiting:
            br_waiting.preferred_text = raw_body
            session.commit()

            try:
                start_dt, end_dt, event_id = schedule_calendar_event(
                    shop=shop,
                    phone_number=from_number,
                    estimate_id=br_waiting.estimate_id,
                    requested_text=raw_body,
                )
            except ValueError:
                twiml.message(
                    "I couldn't understand that date/time. Try again like:\n"
                    '"Tuesday at 2pm" or "Tomorrow at 10am".'
                )
                return Response(str(twiml), media_type="application/xml")
            except Exception as e:
                logger.exception(f"Calendar booking failed: {e}")
                twiml.message(
                    "That time isn't available or booking isn't configured yet. "
                    "Please suggest another time or call the shop."
                )
                return Response(str(twiml), media_type="application/xml")

            br_waiting.step = "completed"
            br_waiting.calendar_event_id = event_id
            session.commit()

            twiml.message(format_booking_confirmation(shop, start_dt))
            return Response(str(twiml), media_type="application/xml")

        # -------------------------------------------------------
        # DEFAULT: no media + not in booking flow
        # -------------------------------------------------------
        twiml.message(
            "To get started, please send 1–3 clear photos of the vehicle damage."
        )
        return Response(str(twiml), media_type="application/xml")

    finally:
        session.close()
