import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioRestClient
import openai

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()

# ============================================================
# ENV + OPENAI (NON-CRASHING)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("WARNING: OPENAI_API_KEY missing – AI features disabled.")

# ============================================================
# DB (NON-CRASHING INIT)
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
engine = None
SessionLocal = None
Base = declarative_base()

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print("Database engine initialized.")
    except Exception as e:
        print("ERROR initializing DB engine:", e)
        engine = None
        SessionLocal = None
else:
    print("WARNING: DATABASE_URL missing – DB persistence disabled.")


class EstimateSession(Base):
    __tablename__ = "estimate_sessions"

    id = Column(Integer, primary_key=True, index=True)
    shop_token = Column(String(100), index=True)
    phone = Column(String(50), index=True)
    analysis_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print("ERROR creating tables:", e)

# ============================================================
# MULTI-SHOP CONFIG (NON-CRASHING)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: Dict[str, Any]
    hours: Dict[str, Any]


def load_shops() -> Dict[str, ShopConfig]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        print("WARNING: SHOPS_JSON missing – no shops configured.")
        return {}
    try:
        data = json.loads(raw)
        return {shop["webhook_token"]: ShopConfig(**shop) for shop in data}
    except Exception as e:
        print("ERROR parsing SHOPS_JSON:", e)
        return {}


shops = load_shops()

# ============================================================
# GOOGLE CALENDAR (ON-DEMAND, GUARDED)
# ============================================================

def get_calendar_service():
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        print("WARNING: GOOGLE_SERVICE_ACCOUNT_JSON missing – calendar disabled.")
        return None
    try:
        info = json.loads(raw_json)
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        print("ERROR initializing Google Calendar:", e)
        return None

# ============================================================
# TWILIO OUTBOUND CLIENT (NON-CRASHING)
# ============================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

twilio_client: Optional[TwilioRestClient] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        print("ERROR initializing Twilio client:", e)
        twilio_client = None
else:
    print("WARNING: Twilio credentials missing – outbound SMS disabled.")

# ============================================================
# UTILS
# ============================================================

def safe_json_parse(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        text = text.lstrip("json").strip()
    return json.loads(text)


def get_db_session():
    if SessionLocal is None:
        return None
    try:
        return SessionLocal()
    except Exception as e:
        print("ERROR opening DB session:", e)
        return None


def save_analysis(shop_token: str, phone: str, analysis: Dict[str, Any]) -> None:
    db = get_db_session()
    if db is None:
        print("DB unavailable – skipping save_analysis.")
        return
    try:
        rec = EstimateSession(
            shop_token=shop_token,
            phone=phone,
            analysis_json=json.dumps(analysis),
        )
        db.add(rec)
        db.commit()
    except Exception as e:
        print("ERROR in save_analysis:", e)
    finally:
        db.close()


def get_latest_analysis(shop_token: str, phone: str) -> Optional[Dict[str, Any]]:
    db = get_db_session()
    if db is None:
        print("DB unavailable – get_latest_analysis returns None.")
        return None
    try:
        rec = (
            db.query(EstimateSession)
            .filter(
                EstimateSession.shop_token == shop_token,
                EstimateSession.phone == phone,
            )
            .order_by(EstimateSession.created_at.desc())
            .first()
        )
        if not rec:
            return None
        return json.loads(rec.analysis_json)
    except Exception as e:
        print("ERROR in get_latest_analysis:", e)
        return None
    finally:
        db.close()


def send_sms(to_number: str, body: str) -> None:
    if not twilio_client or not TWILIO_FROM_NUMBER:
        print("Twilio not configured – cannot send SMS.")
        return
    try:
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=to_number,
        )
    except Exception as e:
        print("ERROR sending SMS:", e)

# ============================================================
# AI HELPERS
# ============================================================

def can_use_ai() -> bool:
    return bool(OPENAI_API_KEY)


def analyze_damage(image_url: str, shop: ShopConfig) -> Dict[str, Any]:
    if not can_use_ai():
        raise RuntimeError("AI not configured")

    pricing = json.dumps(shop.pricing, indent=2)

    prompt = f"""
You are an elite collision damage estimator in Ontario, Canada.

Always describe from DRIVER'S point of view:
- driver side = left
- passenger side = right

Use this pricing:
{pricing}

Return ONLY JSON:
- damage_summary (string)
- areas (string[])
- damage_types (string[])
- severity ("minor" | "moderate" | "severe")
- recommended_labor_hours (number)
- estimated_cost_min (number)
- estimated_cost_max (number)
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a professional auto body collision estimator. Output only JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )
    raw = resp["choices"][0]["message"]["content"]
    return safe_json_parse(raw)


def parse_booking(user_text: str, analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not can_use_ai():
        return None

    summary = analysis.get("damage_summary", "") if analysis else ""
    prompt = f"""
Convert this into booking info.

Message:
\"\"\"{user_text}\"\"\"

Damage summary:
\"\"\"{summary}\"\"\"

Return ONLY JSON:
- name
- phone
- email
- datetime_iso (America/Toronto)
- notes
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = resp["choices"][0]["message"]["content"]
        data = safe_json_parse(raw)
        if not data.get("datetime_iso"):
            return None
        return data
    except Exception as e:
        print("ERROR in parse_booking:", e)
        return None

# ============================================================
# TEXT BUILDERS
# ============================================================

def format_damage_text(shop: ShopConfig, a: Dict[str, Any]) -> str:
    return (
        f"🔍 AI Damage Analysis — {shop.name}\n\n"
        f"{a['damage_summary']}\n\n"
        f"Areas: {', '.join(a.get('areas', []))}\n"
        f"Damage Types: {', '.join(a.get('damage_types', []))}\n"
        f"Severity: {a.get('severity', '').title()}\n\n"
        "If this looks correct, reply 1.\n"
        "If you’d like to upload more photos, reply 2."
    )


def format_full_estimate_text(shop: ShopConfig, a: Dict[str, Any]) -> str:
    eid = str(uuid.uuid4())
    return (
        f"📘 Full Estimate (ID: {eid})\n\n"
        f"Estimated Cost: ${a['estimated_cost_min']} – ${a['estimated_cost_max']} CAD\n"
        f"Labor: {a['recommended_labor_hours']} hours\n\n"
        "This is a visual preliminary estimate.\n\n"
        "To book an appointment, reply with your name, phone, email, and preferred date/time."
    )

# ============================================================
# CALENDAR
# ============================================================

def create_calendar_event(shop: ShopConfig, booking: Dict[str, Any], analysis: Optional[Dict[str, Any]]) -> None:
    service = get_calendar_service()
    if service is None:
        print("Calendar not available – skipping event creation.")
        return

    lines = [
        f"Name: {booking.get('name', '')}",
        f"Phone: {booking.get('phone', '')}",
        f"Email: {booking.get('email', '')}",
        "",
        "Customer Notes:",
        booking.get("notes", ""),
    ]

    if analysis:
        lines.extend([
            "",
            "AI Estimate:",
            analysis.get("damage_summary", ""),
            f"Areas: {', '.join(analysis.get('areas', []))}",
            f"Damage Types: {', '.join(analysis.get('damage_types', []))}",
            f"Severity: {analysis.get('severity', '')}",
            f"Cost: ${analysis.get('estimated_cost_min')} – ${analysis.get('estimated_cost_max')}",
            f"Hours: {analysis.get('recommended_labor_hours')}",
        ])

    desc = "\n".join(lines)

    start = datetime.fromisoformat(booking["datetime_iso"])
    end = start + timedelta(hours=1)

    event = {
        "summary": f"AI Estimate — {booking.get('name', 'Customer')}",
        "description": desc,
        "start": {"dateTime": start.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/Toronto"},
    }

    try:
        service.events().insert(
            calendarId=shop.calendar_id,
            body=event,
        ).execute()
    except Exception as e:
        print("ERROR creating calendar event:", e)

# ============================================================
# BACKGROUND TASKS
# ============================================================

def bg_process_image(shop_token: str, shop: ShopConfig, from_phone: str, media_url: str):
    try:
        if not can_use_ai():
            send_sms(from_phone, "Our AI is temporarily unavailable. Please try again later.")
            return
        analysis = analyze_damage(media_url, shop)
        save_analysis(shop_token, from_phone, analysis)
        send_sms(from_phone, format_damage_text(shop, analysis))
    except Exception as e:
        print("Background image error:", e)
        send_sms(from_phone, "⚠️ There was an issue analyzing your photo. Please try another angle or a clearer image.")


def bg_process_booking(shop: ShopConfig, from_phone: str, text: str, last_analysis: Optional[Dict[str, Any]]):
    try:
        booking = parse_booking(text, last_analysis)
        if not booking:
            send_sms(from_phone, "⚠️ I couldn't understand your booking. Please resend name, phone, email, and preferred date/time.")
            return
        create_calendar_event(shop, booking, last_analysis)
        send_sms(from_phone, "✅ Your appointment has been booked. The shop will contact you if any changes are needed.")
    except Exception as e:
        print("Background booking error:", e)
        send_sms(from_phone, "⚠️ There was a problem booking your appointment. Please try again or call the shop directly.")

# ============================================================
# MAIN WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()

    token = request.query_params.get("token")
    shop = shops.get(token)
    if not shop:
        msg = MessagingResponse()
        msg.message("This shop is not configured correctly. Please contact the shop directly.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    body_raw: str = form.get("Body", "") or ""
    body = body_raw.strip()
    lower = body.lower()
    media_url = form.get("MediaUrl0")
    from_phone = form.get("From", "").strip() or "unknown"

    # Greeting
    if not media_url and lower in ("", "hi", "hello", "hey", "start"):
        msg = MessagingResponse()
        msg.message(
            f"👋 Welcome to {shop.name}!\n\n"
            "Please send 1–3 clear photos of the damage for an instant AI estimate."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    # Image received → background AI
    if media_url:
        background_tasks.add_task(bg_process_image, token, shop, from_phone, media_url)
        msg = MessagingResponse()
        msg.message(
            "📸 Thanks! We received your photos.\n\n"
            "Our AI is analyzing the damage now – you'll get your estimate by text shortly."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    # Confirm analysis → full estimate
    if lower == "1":
        last = get_latest_analysis(token, from_phone)
        msg = MessagingResponse()
        if not last:
            msg.message("I couldn't find your last estimate. Please resend your photos.")
        else:
            msg.message(format_full_estimate_text(shop, last))
        return PlainTextResponse(str(msg), media_type="application/xml")

    # Ask for more photos
    if lower == "2":
        msg = MessagingResponse()
        msg.message("No problem! Please send a few more photos of the damage.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    # Booking → background
    last = get_latest_analysis(token, from_phone)
    background_tasks.add_task(bg_process_booking, shop, from_phone, body_raw, last)

    msg = MessagingResponse()
    msg.message(
        "📅 Thanks! We’re processing your booking request now.\n"
        "You’ll receive a confirmation text shortly."
    )
    return PlainTextResponse(str(msg), media_type="application/xml")
