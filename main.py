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
# Environment
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
# OpenAI client
# ============================================================

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing – AI estimator disabled.")
    openai_client: Optional[OpenAI] = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Database models
# ============================================================


class DamageEstimate(Base):
    __tablename__ = "damage_estimates"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    shop_id = Column(String, index=True)
    phone_number = Column(String, index=True)

    user_message = Column(Text)
    media_urls = Column(Text)  # JSON list

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
    preferred_text = Column(Text, nullable=True)
    calendar_event_id = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)

# ============================================================
# Shop config
# ============================================================


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str  # ?token=...
    calendar_id: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    hours: Optional[Dict[str, Any]] = None


def load_shops() -> Dict[str, Shop]:
    data = json.loads(SHOPS_JSON)
    mapping: Dict[str, Shop] = {}
    for raw in data:
        shop = Shop(**raw)
        mapping[shop.webhook_token] = shop
    return mapping


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# VIN helpers
# ============================================================

VIN_REGEX = r"\b([A-HJ-NPR-Z0-9]{17})\b"


def extract_vin(text: str) -> Optional[str]:
    cleaned = text.replace(" ", "").upper()
    m = re.search(VIN_REGEX, cleaned)
    return m.group(1) if m else None


def decode_vin(vin: str) -> Dict[str, Any]:
    try:
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}?format=json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        result = r.json()["Results"][0]
        return {
            "vin": vin,
            "year": result.get("ModelYear"),
            "make": result.get("Make"),
            "model": result.get("Model"),
            "trim": result.get("Trim"),
            "body_class": result.get("BodyClass"),
            "engine": result.get("EngineCylinders"),
            "fuel_type": result.get("FuelTypePrimary"),
            "series": result.get("Series"),
        }
    except Exception as e:
        logger.warning(f"VIN decode failed: {e}")
        return {"vin": vin, "error": "decode_failed"}


# ============================================================
# Google Calendar
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]
_calendar_service = None


def get_calendar_service():
    global _calendar_service
    if _calendar_service:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def schedule_calendar_event(
    shop: Shop,
    phone_number: str,
    estimate: DamageEstimate,
    preferred_text: str,
):
    if not shop.calendar_id:
        raise RuntimeError("Calendar not configured for this shop")

    try:
        dt = date_parser.parse(preferred_text, fuzzy=True)
    except Exception as e:
        logger.warning(f"Failed to parse datetime from '{preferred_text}': {e}")
        raise ValueError("invalid_datetime")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    else:
        dt = dt.astimezone(LOCAL_TZ)

    start = dt
    end = start + timedelta(hours=1)

    service = get_calendar_service()

    # Simple conflict check
    events_result = (
        service.events()
        .list(
            calendarId=shop.calendar_id,
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
        )
        .execute()
    )
    if events_result.get("items"):
        raise RuntimeError("slot_unavailable")

    description_lines = [
        f"AI damage estimate SMS lead for {shop.name}",
        f"From: {phone_number}",
        f"Estimate ID: {estimate.id}",
        "",
        f"Original message:\n{estimate.user_message or ''}",
        "",
        "Media URLs:",
    ]
    try:
        media_list = json.loads(estimate.media_urls or "[]")
    except Exception:
        media_list = []
    for u in media_list:
        description_lines.append(f"- {u}")
    description_lines.append("")
    description_lines.append(f"Severity: {estimate.severity}")
    description_lines.append(
        f"Estimated cost: ${estimate.estimated_cost_min:,.0f} – ${estimate.estimated_cost_max:,.0f}"
    )

    body = {
        "summary": "AI Collision Estimate Appointment",
        "description": "\n".join(description_lines),
        "start": {"dateTime": start.isoformat(), "timeZone": str(LOCAL_TZ)},
        "end": {"dateTime": end.isoformat(), "timeZone": str(LOCAL_TZ)},
    }

    created = service.events().insert(calendarId=shop.calendar_id, body=body).execute()
    return start, end, created.get("id")


# ============================================================
# AI estimator
# ============================================================


def build_prompt(shop: Shop, user_text: str, vin_data: Optional[Dict[str, Any]]) -> str:
    return f"""
You are a professional auto body estimator in Ontario, Canada (year 2025).

IMPORTANT LEFT/RIGHT RULE:
- When describing vehicle sides, use DRIVER perspective.
- Left = DRIVER side.
- Right = PASSENGER side.

Use the photo, VIN data (if available), and user text to estimate visible collision damage.

Return a realistic collision-repair estimate in Ontario bodyshop pricing, based ONLY on visible damage.

VEHICLE (from VIN, if present):
{json.dumps(vin_data or {}, indent=2)}

CUSTOM SHOP PRICING (if provided):
{json.dumps(shop.pricing or {}, indent=2)}

CUSTOMER MESSAGE:
{user_text}

Respond ONLY with valid JSON in this exact schema:

{{
  "severity": "minor | moderate | severe | total_loss",
  "estimated_cost_min": 0,
  "estimated_cost_max": 0,
  "areas": ["rear bumper", "trunk lid"],
  "damage_types": ["dent", "crack", "paint damage"],
  "notes": "short explanation in plain English",
  "recommendation": "short recommendation, 1–2 sentences"
}}
""".strip()


def call_ai_estimator(
    image_url: str, shop: Shop, user_text: str, vin_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not openai_client:
        raise RuntimeError("openai_not_configured")

    prompt = build_prompt(shop, user_text, vin_data)

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a collision estimator. Return ONLY JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.15,
            max_tokens=500,
        )
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise RuntimeError("openai_request_failed")

    raw = completion.choices[0].message.content or ""

    try:
        return json.loads(raw)
    except Exception:
        # Try to salvage JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            logger.error(f"AI response not JSON: {raw!r}")
            raise RuntimeError("invalid_ai_json")
        try:
            return json.loads(raw[start : end + 1])
        except Exception as e:
            logger.error(f"Failed to parse JSON from AI: {e} | raw={raw!r}")
            raise RuntimeError("invalid_ai_json")


# ============================================================
# SMS helpers
# ============================================================


def sms_for_estimate(shop: Shop, result: Dict[str, Any], estimate_id: str) -> str:
    areas = ", ".join(result.get("areas") or [])
    damage = ", ".join(result.get("damage_types") or [])
    return (
        f"AI Damage Estimate – {shop.name}\n\n"
        f"Severity: {result.get('severity')}\n"
        f"Estimated Cost (Ontario): "
        f"${result.get('estimated_cost_min', 0):,.0f} – ${result.get('estimated_cost_max', 0):,.0f}\n\n"
        f"Areas: {areas or 'N/A'}\n"
        f"Damage Types: {damage or 'N/A'}\n\n"
        "This is a visual estimate only, not a final bill.\n"
        "Reply 1 to book a repair appointment.\n\n"
        f"Estimate ID: {estimate_id}"
    )


def sms_for_booking_prompt() -> str:
    return (
        "Great! To book your visit, reply with a day & time that works for you.\n"
        "Examples:\n"
        "- Tomorrow at 3pm\n"
        "- Friday 10am\n"
        "- Next Tuesday afternoon"
    )


def sms_for_booking_confirm(shop: Shop, start: datetime) -> str:
    local = start.astimezone(LOCAL_TZ)
    return (
        f"You're booked at {shop.name} on "
        f"{local.strftime('%A, %B %d at %I:%M %p')}.\n"
        "You'll get a calendar confirmation from the shop as well."
    )


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")

    shop = SHOPS_BY_TOKEN[token]

    form = await request.form()
    from_number = form.get("From")
    body_raw = form.get("Body") or ""
    body = body_raw.strip()
    media_count = int(form.get("NumMedia") or "0")

    twiml = MessagingResponse()
    db = SessionLocal()

    try:
        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        # --------------------------
        # New estimate with photo(s)
        # --------------------------
        if media_count > 0:
            media_urls: List[str] = []
            for i in range(media_count):
                key = f"MediaUrl{i}"
                url = form.get(key)
                if url:
                    media_urls.append(url)

            # Call AI on the first image
            image_url = media_urls[0]

            try:
                result = call_ai_estimator(image_url, shop, body_raw, vin_data)
            except RuntimeError as e:
                logger.error(f"Estimator failed: {e}")
                twiml.message(
                    "Error generating estimate right now. Please try again in a few minutes "
                    "or call the shop directly."
                )
                return Response(str(twiml), media_type="application/xml")

            estimate_id = str(uuid.uuid4())

            estimate = DamageEstimate(
                id=estimate_id,
                shop_id=shop.id,
                phone_number=from_number,
                user_message=body_raw,
                media_urls=json.dumps(media_urls),
                vin=vin,
                vin_decoded=json.dumps(vin_data) if vin_data else None,
                severity=result.get("severity"),
                estimated_cost_min=float(result.get("estimated_cost_min") or 0),
                estimated_cost_max=float(result.get("estimated_cost_max") or 0),
                vehicle_info=body_raw,
                ai_raw_json=json.dumps(result),
                ai_summary_text=result.get("notes") or "",
            )
            db.add(estimate)
            db.commit()

            booking = BookingRequest(
                id=str(uuid.uuid4()),
                shop_id=shop.id,
                phone_number=from_number,
                estimate_id=estimate_id,
                step="awaiting_choice",
            )
            db.add(booking)
            db.commit()

            twiml.message(sms_for_estimate(shop, result, estimate_id))
            return Response(str(twiml), media_type="application/xml")

        # --------------------------------------------------
        # User replying "1" to book appointment
        # --------------------------------------------------
        if body.lower() in {"1", "book", "yes"}:
            booking = (
                db.query(BookingRequest)
                .filter(
                    BookingRequest.shop_id == shop.id,
                    BookingRequest.phone_number == from_number,
                )
                .order_by(BookingRequest.created_at.desc())
                .first()
            )
            if not booking or not booking.estimate_id:
                twiml.message("Please send photos of the damage first so we can create an estimate.")
                return Response(str(twiml), media_type="application/xml")

            booking.step = "awaiting_datetime"
            db.commit()

            twiml.message(sms_for_booking_prompt())
            return Response(str(twiml), media_type="application/xml")

        # --------------------------------------------------
        # User sending a date/time after choosing to book
        # --------------------------------------------------
        existing = (
            db.query(BookingRequest)
            .filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == from_number,
                BookingRequest.step == "awaiting_datetime",
            )
            .order_by(BookingRequest.created_at.desc())
            .first()
        )

        if existing:
            existing.preferred_text = body_raw
            db.commit()

            estimate = db.query(DamageEstimate).filter(DamageEstimate.id == existing.estimate_id).first()

            if not estimate:
                twiml.message("We couldn't find your estimate. Please resend your photos.")
                return Response(str(twiml), media_type="application/xml")

            try:
                start, _end, event_id = schedule_calendar_event(
                    shop=shop,
                    phone_number=from_number,
                    estimate=estimate,
                    preferred_text=body_raw,
                )
            except ValueError:
                twiml.message(
                    "Sorry, we couldn't understand that date/time. "
                    "Please try again like 'Tomorrow at 3pm' or 'Friday 10am'."
                )
                return Response(str(twiml), media_type="application/xml")
            except RuntimeError as e:
                logger.error(f"Calendar booking error: {e}")
                # Don't lose the lead, just tell them the shop will follow up
                twiml.message(
                    "Thanks! We've received your preferred time. "
                    "A team member will confirm your appointment shortly."
                )
                existing.step = "awaiting_manual_confirm"
                db.commit()
                return Response(str(twiml), media_type="application/xml")

            existing.step = "completed"
            existing.calendar_event_id = event_id
            db.commit()

            twiml.message(sms_for_booking_confirm(shop, start))
            return Response(str(twiml), media_type="application/xml")

        # --------------------------------------------------
        # Default fallback
        # --------------------------------------------------
        twiml.message(
            "Welcome to our AI damage estimate service.\n\n"
            "Please send 1–3 clear photos of the vehicle damage (rear, front, side). "
            "You'll receive an instant rough estimate and the option to book a visit."
        )
        return Response(str(twiml), media_type="application/xml")

    finally:
        db.close()
