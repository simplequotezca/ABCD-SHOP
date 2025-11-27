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

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI — v1 safe init
# ============================================================

client = OpenAI()

# ============================================================
# SQLAlchemy Models
# ============================================================

class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_primary_key=True, index=True)
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

    step = Column(String, index=True)
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
    shops = {}
    for item in raw:
        shop = Shop(**item)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN DECODER
# ============================================================

VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"   # excludes I, O, Q automatically


def extract_vin(text: str) -> Optional[str]:
    """Find VIN anywhere in user message."""
    match = re.search(VIN_REGEX, text.replace(" ", "").upper())
    return match.group(1) if match else None


def decode_vin(vin: str) -> Dict[str, Any]:
    """Use NHTSA public API to decode VIN (no key needed)."""
    try:
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}?format=json"
        res = requests.get(url, timeout=5)
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
    except Exception:
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

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def schedule_calendar_event(shop, phone_number, estimate_id, requested_text):
    """Parse date/time + book appointment."""
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
        raise RuntimeError("Requested time unavailable")

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

def build_ai_prompt(shop, user_text, vehicle_hint, pricing_hint, vin_data):
    pricing_str = json.dumps(pricing_hint, indent=2) if pricing_hint else "null"
    vin_str = json.dumps(vin_data, indent=2) if vin_data else "{}"

    return f"""
You are an expert collision estimator in Ontario (2025).

CRITICAL LEFT/RIGHT RULE:
- Always interpret left/right based on DRIVER POV sitting in the driver seat.
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

Return ONLY valid JSON using the provided schema.
"""


def call_ai_estimator(image_url, shop, user_text, vehicle_hint, vin_data):
    prompt = build_ai_prompt(
        shop, user_text, vehicle_hint, shop.pricing, vin_data
    )

    completion = client.chat.completions.create(
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
    except:
        start = content.find("{")
        end = content.rfind("}")
        return json.loads(content[start:end + 1])

# ============================================================
# Formatting Messages
# ============================================================

def format_estimate_sms(shop, estimate, estimate_id):
    severity = estimate.get("severity")
    min_cost = estimate.get("estimated_cost_min")
    max_cost = estimate.get("estimated_cost_max")

    areas = ", ".join(estimate.get("areas") or [])
    damages = ", ".join(estimate.get("damage_types") or [])

    cost_str = f"${min_cost:,.0f} – ${max_cost:,.0f}"

    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {severity}\n"
        f"Estimated Cost (Ontario 2025): {cost_str}\n\n"
        f"Areas: {areas}\n"
        f"Damage Types: {damages}\n\n"
        f"This is a visual, preliminary estimate only. Final pricing may vary after an in-person inspection.\n\n"
        f"Reply 1 to schedule an appointment.\n\n"
        f"Estimate ID: {estimate_id}"
    )


def format_booking_confirmation(shop, start_dt):
    date = start_dt.strftime("%A, %B %d")
    time = start_dt.strftime("%I:%M %p").lstrip("0")

    return (
        f"You're booked at {shop.name} on {date} at {time}.\n\n"
        "If you need to make any changes, please contact the shop directly."
    )

# ============================================================
# Twilio Webhook
# ============================================================

app = FastAPI()

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(400, "Missing token")

    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(403, "Invalid token")

    form = await request.form()
    from_number = form.get("From")
    raw_body = form.get("Body") or ""
    body = raw_body.strip()
    num_media = int(form.get("NumMedia") or "0")

    session = SessionLocal()
    twiml = MessagingResponse()

    try:
        # =======================================================
        # VIN Auto-Detection (in ANY message)
        # =======================================================
        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        # =======================================================
        # USER SENDS IMAGES → AI ESTIMATE
        # =======================================================
        if num_media > 0:
            image_url = form.get("MediaUrl0")

            estimate_json = call_ai_estimator(
                image_url, shop, body, body, vin_data
            )

            estimate_id = str(uuid.uuid4())

            session.add(
                DamageEstimate(
                    id=estimate_id,
                    shop_id=shop.id,
                    phone_number=from_number,
                    user_message=raw_body,
                    vin=vin,
                    vin_decoded=json.dumps(vin_data),
                    severity=estimate_json.get("severity"),
                    estimated_cost_min=estimate_json.get("estimated_cost_min"),
                    estimated_cost_max=estimate_json.get("estimated_cost_max"),
                    vehicle_info=raw_body,
                    ai_raw_json=json.dumps(estimate_json),
                )
            )
            session.commit()

            # Start booking flow
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

            twiml.message(format_estimate_sms(shop, estimate_json, estimate_id))
            return Response(content=str(twiml), media_type="application/xml")

        # =======================================================
        # USER REPLIES "1" → ASK FOR DATE/TIME
        # =======================================================
        if body in {"1", "one", "book"}:
            br = (
                session.query(BookingRequest)
                .filter(BookingRequest.shop_id == shop.id,
                        BookingRequest.phone_number == from_number)
                .order_by(BookingRequest.created_at.desc())
                .first()
            )

            if not br:
                twiml.message("Please send photos of the damage to begin.")
                return Response(content=str(twiml), media_type="application/xml")

            br.step = "awaiting_datetime"
            session.commit()

            twiml.message(
                "Great! What day and time works best for you?\n\n"
                "Examples:\n"
                "\"Tuesday at 3pm\"\n"
                "\"Tomorrow at 10am\"\n"
                "\"Friday morning\""
            )
            return Response(content=str(twiml), media_type="application/xml")

        # =======================================================
        # WAITING FOR DATE/TIME → TRY BOOKING
        # =======================================================
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
            br_waiting.preferred_text = body

            try:
                start_dt, end_dt, event_id = schedule_calendar_event(
                    shop, from_number, br_waiting.estimate_id, body
                )
            except ValueError:
                twiml.message(
                    "I couldn't understand that date/time. Try again:\n"
                    "\"Tuesday at 2pm\" or \"Tomorrow at 10am\""
                )
                return Response(content=str(twiml), media_type="application/xml")
            except:
                twiml.message(
                    "That time isn’t available. Please suggest another time."
                )
                return Response(content=str(twiml), media_type="application/xml")

            br_waiting.step = "completed"
            br_waiting.calendar_event_id = event_id
            session.commit()

            twiml.message(format_booking_confirmation(shop, start_dt))
            return Response(content=str(twiml), media_type="application/xml")

        # =======================================================
        # DEFAULT RESPONSE
        # =======================================================
        twiml.message("Please send 1–3 clear photos of the damage to begin.")
        return Response(content=str(twiml), media_type="application/xml")

    finally:
        session.close()
