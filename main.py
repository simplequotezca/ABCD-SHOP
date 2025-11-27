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

LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI Client (SAFE)
# ============================================================

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing — AI estimator disabled.")
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

    step = Column(String, index=True)
    preferred_text = Column(Text)
    calendar_event_id = Column(String)


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
    raw = json.loads(SHOPS_JSON)
    shops = {}
    for entry in raw:
        shop = Shop(**entry)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN Handling
# ============================================================

VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"


def extract_vin(text: str) -> Optional[str]:
    clean = text.replace(" ", "").upper()
    match = re.search(VIN_REGEX, clean)
    return match.group(1) if match else None


def decode_vin(vin: str) -> Dict[str, Any]:
    try:
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}?format=json"
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
        logger.warning(f"VIN decode failed: {e}")
        return {"vin": vin, "error": "decode_failed"}

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
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def schedule_calendar_event(shop: Shop, phone_number: str, estimate_id: str, text: str):
    if not shop.calendar_id:
        raise RuntimeError("Calendar not configured.")

    try:
        dt = date_parser.parse(text, fuzzy=True)
    except Exception:
        raise ValueError("Invalid date/time format")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)

    start = dt
    end = start + timedelta(hours=1)

    service = get_calendar_service()
    events = service.events().list(
        calendarId=shop.calendar_id,
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
    ).execute().get("items", [])

    if events:
        raise RuntimeError("Unavailable")

    body = {
        "summary": "Auto Body Estimate Appointment",
        "description": f"SMS lead from {phone_number}. Estimate ID: {estimate_id}",
        "start": {"dateTime": start.isoformat(), "timeZone": str(LOCAL_TZ)},
        "end": {"dateTime": end.isoformat(), "timeZone": str(LOCAL_TZ)},
    }

    event = service.events().insert(calendarId=shop.calendar_id, body=body).execute()
    return start, end, event.get("id")

# ============================================================
# AI Estimator
# ============================================================

def build_prompt(shop: Shop, user_text: str, vin_data: dict) -> str:
    return f"""
You are an Ontario collision estimator (2025).

LEFT/RIGHT RULE:
- Left = driver side.
- Right = passenger side.

VEHICLE (from VIN if available):
{json.dumps(vin_data or {}, indent=2)}

USER TEXT:
{user_text}

SHOP PRICING:
{json.dumps(shop.pricing or {}, indent=2)}

Return ONLY valid JSON:
{{
  "severity": "...",
  "estimated_cost_min": 0,
  "estimated_cost_max": 0,
  "areas": [],
  "damage_types": [],
  "notes": "",
  "recommendation": ""
}}
"""


def call_ai_estimator(image_url: str, shop: Shop, text: str, vin_data: dict):
    if not openai_client:
        raise RuntimeError("OpenAI not configured")

    prompt = build_prompt(shop, text, vin_data)

    # IMPORTANT FIX: use full vision model gpt-4o
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
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
        max_tokens=500,
    )

    raw = completion.choices[0].message.content

    try:
        return json.loads(raw)
    except:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1:
            raise RuntimeError("Invalid AI JSON")
        return json.loads(raw[s:e+1])

# ============================================================
# SMS Formatting
# ============================================================

def sms_estimate(shop: Shop, result: dict, est_id: str):
    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {result.get('severity')}\n"
        f"Estimated Cost: ${result.get('estimated_cost_min'):,.0f} – ${result.get('estimated_cost_max'):,.0f}\n\n"
        f"Areas: {', '.join(result.get('areas') or [])}\n"
        f"Damage: {', '.join(result.get('damage_types') or [])}\n\n"
        f"Visual estimate only. Reply 1 to book an appointment.\n\n"
        f"ID: {est_id}"
    )


def sms_confirmation(shop: Shop, dt: datetime):
    return f"You're booked at {shop.name} on {dt.strftime('%A %B %d at %I:%M %p')}."

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(403, "Invalid token")

    shop = SHOPS_BY_TOKEN[token]

    form = await request.form()
    number = form.get("From")
    raw = form.get("Body") or ""
    body = raw.strip()
    media_count = int(form.get("NumMedia") or "0")

    session = SessionLocal()
    tw = MessagingResponse()

    try:
        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        # -------------------------
        # PHOTO RECEIVED
        # -------------------------
        if media_count > 0:
            url = form.get("MediaUrl0")

            try:
                result = call_ai_estimator(url, shop, raw, vin_data)
            except Exception as e:
                logger.error(f"AI error: {e}")
                tw.message("Error generating estimate. Try again shortly.")
                return Response(str(tw), media_type="application/xml")

            est_id = str(uuid.uuid4())

            session.add(DamageEstimate(
                id=est_id,
                shop_id=shop.id,
                phone_number=number,
                user_message=raw,
                vin=vin,
                vin_decoded=json.dumps(vin_data) if vin_data else None,
                severity=result.get("severity"),
                estimated_cost_min=result.get("estimated_cost_min"),
                estimated_cost_max=result.get("estimated_cost_max"),
                vehicle_info=raw,
                ai_raw_json=json.dumps(result),
            ))
            session.commit()

            session.add(BookingRequest(
                id=str(uuid.uuid4()),
                shop_id=shop.id,
                phone_number=number,
                estimate_id=est_id,
                step="choice",
            ))
            session.commit()

            tw.message(sms_estimate(shop, result, est_id))
            return Response(str(tw), media_type="application/xml")

        # -------------------------
        # USER REPLIED "1"
        # -------------------------
        if body.lower() in {"1", "book", "one"}:
            br = session.query(BookingRequest).filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == number,
            ).order_by(BookingRequest.created_at.desc()).first()

            if not br:
                tw.message("Please send photos first to get an estimate.")
                return Response(str(tw), media_type="application/xml")

            br.step = "awaiting_datetime"
            session.commit()

            tw.message(
                "What day/time works best?\n"
                "Examples:\n"
                "- Tomorrow at 3pm\n"
                "- Friday 10am\n"
                "- Tuesday morning"
            )
            return Response(str(tw), media_type="application/xml")

        # -------------------------
        # USER PROVIDING DATE/TIME
        # -------------------------
        br = session.query(BookingRequest).filter(
            BookingRequest.shop_id == shop.id,
            BookingRequest.phone_number == number,
            BookingRequest.step == "awaiting_datetime",
        ).order_by(BookingRequest.created_at.desc()).first()

        if br:
            br.preferred_text = raw
            session.commit()

            try:
                start, end, event_id = schedule_calendar_event(
                    shop, number, br.estimate_id, raw
                )
            except ValueError:
                tw.message("Couldn't understand the time. Try again.")
                return Response(str(tw), media_type="application/xml")
            except Exception:
                tw.message("That time isn't available. Suggest another.")
                return Response(str(tw), media_type="application/xml")

            br.step = "completed"
            br.calendar_event_id = event_id
            session.commit()

            tw.message(sms_confirmation(shop, start))
            return Response(str(tw), media_type="application/xml")

        # -------------------------
        # DEFAULT
        # -------------------------
        tw.message("Please send 1–3 clear photos of the damage.")
        return Response(str(tw), media_type="application/xml")

    finally:
        session.close()
