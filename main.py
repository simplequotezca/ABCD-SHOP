import os
import json
import uuid
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

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

# ============================================================
# Database Setup
# ============================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI Client — v1 Correct Usage
# ============================================================

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing – AI calls will fail")
    openai_client = None
else:
    openai_client = OpenAI()   # v1 client reads env automatically

# ============================================================
# SQLAlchemy Models
# ============================================================

class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    shop_id = Column(String)
    phone_number = Column(String)
    user_message = Column(Text)

    vin = Column(String)
    vin_decoded = Column(Text)

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

    shop_id = Column(String)
    phone_number = Column(String)
    estimate_id = Column(String)

    step = Column(String)  # choice → awaiting_datetime → completed
    preferred_text = Column(Text)
    calendar_event_id = Column(String)


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


def load_shops():
    raw = json.loads(SHOPS_JSON)
    result = {}
    for s in raw:
        shop = Shop(**s)
        result[shop.webhook_token] = shop
    return result


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN Decoder
# ============================================================

VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"


def extract_vin(text: str) -> Optional[str]:
    compact = text.replace(" ", "").upper()
    found = re.search(VIN_REGEX, compact)
    return found.group(1) if found else None


def decode_vin(vin: str) -> Dict[str, Any]:
    try:
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}?format=json"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()["Results"][0]
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
        logger.warning(f"VIN decode failed: {e}")
        return {"vin": vin, "error": "VIN decoding failed"}

# ============================================================
# Google Calendar Integration
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]
_calendar_service = None


def get_calendar_service():
    global _calendar_service
    if _calendar_service:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON missing")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def schedule_calendar_event(shop: Shop, phone_number: str, estimate_id: str, text: str):
    if not shop.calendar_id:
        raise RuntimeError("Shop missing calendar_id")

    try:
        dt = date_parser.parse(text, fuzzy=True)
    except Exception:
        raise ValueError("Invalid date/time format")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)

    start = dt
    end = dt + timedelta(hours=1)

    cal = get_calendar_service()
    conflicts = cal.events().list(
        calendarId=shop.calendar_id,
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
    ).execute().get("items", [])

    if conflicts:
        raise RuntimeError("Requested time unavailable")

    event = cal.events().insert(
        calendarId=shop.calendar_id,
        body={
            "summary": "Auto Body Estimate Appointment",
            "description": f"SMS lead from {phone_number} — Estimate {estimate_id}",
            "start": {"dateTime": start.isoformat(), "timeZone": str(LOCAL_TZ)},
            "end": {"dateTime": end.isoformat(), "timeZone": str(LOCAL_TZ)},
        },
    ).execute()

    return start, end, event["id"]

# ============================================================
# AI Damage Estimator
# ============================================================

def build_ai_prompt(shop: Shop, user_text: str, pricing: Dict[str, Any], vin_data: Dict[str, Any]):
    return f"""
You are an expert collision estimator in Ontario, Canada (2025).

IMPORTANT: interpret left/right from the DRIVER'S perspective (driver = left).

VEHICLE DETAILS:
{json.dumps(vin_data or {}, indent=2)}

USER MESSAGE:
{user_text}

PRICING:
{json.dumps(pricing or {}, indent=2)}

Return ONLY valid JSON:
{{
  "severity": "...",
  "estimated_cost_min": number,
  "estimated_cost_max": number,
  "areas": [],
  "damage_types": [],
  "notes": "",
  "recommendation": ""
}}
"""


def run_ai_estimator(image_url: str, shop: Shop, text: str, vin_data: Optional[Dict[str, Any]]):
    if not openai_client:
        raise RuntimeError("AI unavailable")

    prompt = build_ai_prompt(shop, text, shop.pricing, vin_data)

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

    raw = completion.choices[0].message.content

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        return json.loads(raw[start:end + 1])

# ============================================================
# SMS Reply Formatting
# ============================================================

def format_estimate(shop: Shop, est: Dict[str, Any], estimate_id: str):
    severity = est.get("severity", "Unknown")
    mn = est.get("estimated_cost_min")
    mx = est.get("estimated_cost_max")

    try:
        cost = f"${mn:,.0f} – ${mx:,.0f}"
    except Exception:
        cost = "N/A"

    areas = ", ".join(est.get("areas") or [])
    dmg = ", ".join(est.get("damage_types") or [])

    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {severity}\n"
        f"Estimated Cost: {cost}\n\n"
        f"Areas: {areas}\n"
        f"Damage Types: {dmg}\n\n"
        "Reply 1 to schedule an appointment.\n\n"
        f"Estimate ID: {estimate_id}"
    )


def format_booking_confirm(shop: Shop, dt: datetime):
    return (
        f"You're booked at {shop.name} on "
        f"{dt.strftime('%A, %B %d at %I:%M %p')}."
    )

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(400, "Missing token")

    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(403, "Invalid shop token")

    form = await request.form()

    from_number = form.get("From")
    raw_body = form.get("Body") or ""
    text = raw_body.strip()
    num_media = int(form.get("NumMedia") or 0)

    session = SessionLocal()
    tw = MessagingResponse()

    try:
        # VIN auto-detection
        vin = extract_vin(text)
        vin_data = decode_vin(vin) if vin else None

        # =====================================================
        # CASE 1 — Photo Uploaded → Run AI Estimator
        # =====================================================
        if num_media > 0:
            image_url = form.get("MediaUrl0")
            if not image_url:
                tw.message("Couldn't read the photo — please resend.")
                return Response(str(tw), media_type="application/xml")

            try:
                est = run_ai_estimator(image_url, shop, raw_body, vin_data)
            except Exception as e:
                logger.error(f"AI error: {e}")
                tw.message("Error generating estimate. Try again shortly.")
                return Response(str(tw), media_type="application/xml")

            estimate_id = str(uuid.uuid4())

            # Save estimate
            session.add(
                DamageEstimate(
                    id=estimate_id,
                    shop_id=shop.id,
                    phone_number=from_number,
                    user_message=raw_body,
                    vin=vin,
                    vin_decoded=json.dumps(vin_data) if vin_data else None,
                    severity=est.get("severity"),
                    estimated_cost_min=est.get("estimated_cost_min"),
                    estimated_cost_max=est.get("estimated_cost_max"),
                    vehicle_info=raw_body,
                    ai_raw_json=json.dumps(est),
                )
            )
            session.commit()

            # Create booking session
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

            tw.message(format_estimate(shop, est, estimate_id))
            return Response(str(tw), media_type="application/xml")

        # =====================================================
        # CASE 2 — Reply "1" → Ask for date/time
        # =====================================================
        if text.lower() in {"1", "book"}:
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
                tw.message("Send photos of the damage first.")
                return Response(str(tw), media_type="application/xml")

            br.step = "awaiting_datetime"
            session.commit()

            tw.message(
                "Great — what date & time works?\n"
                "Examples:\n"
                "• Tuesday 3pm\n"
                "• Tomorrow 10am\n"
                "• Friday morning"
            )
            return Response(str(tw), media_type="application/xml")

        # =====================================================
        # CASE 3 — Awaiting date/time → Try to schedule
        # =====================================================
        br = (
            session.query(BookingRequest)
            .filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == from_number,
                BookingRequest.step == "awaiting_datetime",
            )
            .order_by(BookingRequest.created_at.desc())
            .first()
        )

        if br:
            br.preferred_text = raw_body
            session.commit()

            try:
                start, end, event_id = schedule_calendar_event(
                    shop, from_number, br.estimate_id, raw_body
                )
            except ValueError:
                tw.message("Couldn't understand that time — try again.")
                return Response(str(tw), media_type="application/xml")
            except Exception:
                tw.message("That time isn't available — try another.")
                return Response(str(tw), media_type="application/xml")

            br.step = "completed"
            br.calendar_event_id = event_id
            session.commit()

            tw.message(format_booking_confirm(shop, start))
            return Response(str(tw), media_type="application/xml")

        # =====================================================
        # DEFAULT — No photos, not in booking flow
        # =====================================================
        tw.message("Please send 1–3 photos of the damage to begin.")
        return Response(str(tw), media_type="application/xml")

    finally:
        session.close()
