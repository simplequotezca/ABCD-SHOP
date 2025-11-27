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
SHOPS_JSON = os.getenv("SHOPS_JSON")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON is not set")

# SQLAlchemy base / session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

LOCAL_TZ = ZoneInfo("America/Toronto")

# ============================================================
# OpenAI Client (lazy, safe)
# ============================================================

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Lazily construct the OpenAI client so that any library/config
    issues don't crash the app at import time.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing — AI estimator disabled.")
        raise RuntimeError("Missing OPENAI_API_KEY")

    try:
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise RuntimeError("OpenAI initialization failed") from e


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
    media_urls = Column(Text)  # JSON list of URLs
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


# Try to create tables, but don't crash app if DB is temporarily unreachable
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.error(f"DB init failed (will retry on first request): {e}")


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
    shops: Dict[str, Shop] = {}
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
    if _calendar_service is not None:
        return _calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    _calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _calendar_service


def schedule_calendar_event(
    shop: Shop,
    phone_number: str,
    estimate: Optional[DamageEstimate],
    user_datetime_text: str,
) -> tuple[datetime, datetime, str]:
    if not shop.calendar_id:
        raise RuntimeError("Calendar not configured for this shop.")

    # Parse natural language time
    try:
        dt = date_parser.parse(user_datetime_text, fuzzy=True)
    except Exception as e:
        logger.warning(f"Failed to parse datetime from '{user_datetime_text}': {e}")
        raise ValueError("Invalid date/time format")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    else:
        dt = dt.astimezone(LOCAL_TZ)

    start = dt
    end = start + timedelta(hours=1)

    service = get_calendar_service()

    # Check for conflicts
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
    events = events_result.get("items", [])
    if events:
        raise RuntimeError("Requested time slot is unavailable")

    # Build rich description with all info + image URLs
    description_lines: List[str] = [
        f"SMS lead from {phone_number}",
    ]
    if estimate is not None:
        description_lines.extend(
            [
                "",
                f"Estimate ID: {estimate.id}",
                f"Severity: {estimate.severity}",
                f"Estimated cost: ${estimate.estimated_cost_min:,.0f} – ${estimate.estimated_cost_max:,.0f}",
                "",
                "Original customer message:",
                estimate.user_message or "",
            ]
        )
        try:
            media_urls = json.loads(estimate.media_urls or "[]")
        except Exception:
            media_urls = []
        if media_urls:
            description_lines.append("")
            description_lines.append("Photo URLs:")
            description_lines.extend(media_urls)

    description = "\n".join(description_lines)

    event_body = {
        "summary": f"Auto Body Estimate – {phone_number}",
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": str(LOCAL_TZ)},
        "end": {"dateTime": end.isoformat(), "timeZone": str(LOCAL_TZ)},
    }

    event = service.events().insert(calendarId=shop.calendar_id, body=event_body).execute()
    return start, end, event.get("id")


# ============================================================
# AI Estimator
# ============================================================

def build_prompt(shop: Shop, user_text: str, vin_data: Optional[dict]) -> str:
    """
    Prompt tuned for Ontario 2025 collision estimating with LEFT/RIGHT rule.
    """
    return f"""
You are a professional auto body damage estimator in Ontario, Canada (year 2025).

Use these rules:
- "Left" side ALWAYS means DRIVER side.
- "Right" side ALWAYS means PASSENGER side.
- Give realistic collision repair costs for Ontario body shops (labor, materials, paint, parts).
- Use the shop's pricing if provided.
- Assume this is a visual preliminary estimate, NOT a final bill.

VEHICLE (from VIN if available):
{json.dumps(vin_data or {}, indent=2)}

CUSTOMER MESSAGE (may include extra info about the accident or timing preferences):
{user_text}

SHOP PRICING (if any – can be empty):
{json.dumps(shop.pricing or {}, indent=2)}

Your job:
1. Carefully inspect the image and text.
2. Identify ALL damaged areas (hood, bumper, fender, doors, trunk, roof, lights, etc).
3. Describe typical damage types (scratches, dents, cracks, panel deformation, misalignment, structural concerns).
4. Estimate a realistic cost range for a professional repair in Ontario.

Return ONLY valid JSON in this EXACT format (no extra text):

{{
  "severity": "minor | moderate | severe",
  "estimated_cost_min": 0,
  "estimated_cost_max": 0,
  "areas": ["rear bumper", "trunk lid"],
  "damage_types": ["deep dent", "panel deformation"],
  "notes": "Short explanation of the damage and repair considerations.",
  "recommendation": "Short, clear recommendation for the customer."
}}
"""


def call_ai_estimator(image_url: str, shop: Shop, text: str, vin_data: Optional[dict]) -> dict:
    client = get_openai_client()
    prompt = build_prompt(shop, text, vin_data)

    completion = client.chat.completions.create(
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
        temperature=0.2,
        max_tokens=600,
    )

    raw = completion.choices[0].message.content or ""

    # Try strict JSON first, then fall back to extracting the first {...} block
    try:
        return json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1:
            logger.error(f"AI returned non-JSON content: {raw!r}")
            raise RuntimeError("AI did not return JSON")
        try:
            return json.loads(raw[s: e + 1])
        except Exception as e2:
            logger.error(f"Failed to parse AI JSON: {e2} from {raw!r}")
            raise RuntimeError("AI JSON parse failed") from e2


# ============================================================
# SMS Formatting
# ============================================================

def sms_estimate(shop: Shop, result: dict, est_id: str) -> str:
    areas = ", ".join(result.get("areas") or [])
    damage_types = ", ".join(result.get("damage_types") or [])
    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {result.get('severity')}\n"
        f"Estimated Cost (Ontario 2025): "
        f"${result.get('estimated_cost_min'):,.0f} – ${result.get('estimated_cost_max'):,.0f}\n\n"
        f"Areas: {areas}\n"
        f"Damage Types: {damage_types}\n\n"
        "This is a visual, preliminary estimate – not a final repair bill.\n"
        "Reply 1 to book an in-person appointment.\n\n"
        f"Estimate ID: {est_id}"
    )


def sms_confirmation(shop: Shop, dt: datetime) -> str:
    return (
        f"You're booked at {shop.name} on "
        f"{dt.strftime('%A %B %d at %I:%M %p')}."
    )


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    # -----------------------------
    # Shop resolution via token
    # -----------------------------
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    shop = SHOPS_BY_TOKEN[token]

    form = await request.form()
    number = form.get("From")
    raw_body = form.get("Body") or ""
    body = raw_body.strip()
    media_count = int(form.get("NumMedia") or "0")

    session = SessionLocal()
    tw = MessagingResponse()

    try:
        # Ensure DB tables exist (retry if first init failed earlier)
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            logger.error(f"DB create_all during request failed: {e}")

        vin = extract_vin(body)
        vin_data = decode_vin(vin) if vin else None

        media_urls: List[str] = []
        for i in range(media_count):
            url_key = f"MediaUrl{i}"
            url_val = form.get(url_key)
            if url_val:
                media_urls.append(url_val)

        # ====================================================
        # 1) PHOTO RECEIVED – generate estimate
        # ====================================================
        if media_count > 0:
            image_url = media_urls[0]

            try:
                result = call_ai_estimator(image_url, shop, raw_body, vin_data)
            except Exception as e:
                logger.error(f"AI error: {e}")
                tw.message(
                    "Error generating estimate right now. "
                    "Please try again in a few minutes."
                )
                return Response(str(tw), media_type="application/xml")

            est_id = str(uuid.uuid4())

            estimate = DamageEstimate(
                id=est_id,
                shop_id=shop.id,
                phone_number=number,
                user_message=raw_body,
                vin=vin,
                vin_decoded=json.dumps(vin_data) if vin_data else None,
                severity=result.get("severity"),
                estimated_cost_min=result.get("estimated_cost_min"),
                estimated_cost_max=result.get("estimated_cost_max"),
                vehicle_info=raw_body,
                media_urls=json.dumps(media_urls),
                ai_raw_json=json.dumps(result),
                ai_summary_text=result.get("notes") or "",
            )
            session.add(estimate)
            session.commit()

            # Create/refresh booking workflow state
            booking = BookingRequest(
                id=str(uuid.uuid4()),
                shop_id=shop.id,
                phone_number=number,
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
                    BookingRequest.phone_number == number,
                )
                .order_by(BookingRequest.created_at.desc())
                .first()
            )

            if not booking:
                tw.message("Please send photos of the damage first to get an estimate.")
                return Response(str(tw), media_type="application/xml")

            booking.step = "awaiting_datetime"
            session.commit()

            tw.message(
                "What day and time works best for you?\n"
                "Examples:\n"
                "- Tomorrow at 3pm\n"
                "- Friday 10am\n"
                "- Next Tuesday morning"
            )
            return Response(str(tw), media_type="application/xml")

        # ====================================================
        # 3) USER SENDING DATE/TIME – complete booking
        # ====================================================
        booking = (
            session.query(BookingRequest)
            .filter(
                BookingRequest.shop_id == shop.id,
                BookingRequest.phone_number == number,
                BookingRequest.step == "awaiting_datetime",
            )
            .order_by(BookingRequest.created_at.desc())
            .first()
        )

        if booking:
            booking.preferred_text = raw_body
            session.commit()

            estimate: Optional[DamageEstimate] = None
            if booking.estimate_id:
                estimate = session.query(DamageEstimate).get(booking.estimate_id)

            try:
                start, end, event_id = schedule_calendar_event(
                    shop=shop,
                    phone_number=number,
                    estimate=estimate,
                    user_datetime_text=raw_body,
                )
            except ValueError:
                tw.message(
                    "Sorry, I couldn't understand that date/time. "
                    "Please try something like 'Tomorrow at 3pm' or 'Friday 10am'."
                )
                return Response(str(tw), media_type="application/xml")
            except Exception as e:
                logger.error(f"Calendar scheduling error: {e}")
                tw.message(
                    "That time isn't available or our calendar is busy. "
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
