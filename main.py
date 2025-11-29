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
# BASIC CONFIG
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")
openai.api_key = OPENAI_API_KEY

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Twilio outbound client for background messages ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")  # e.g. "+16473722080"

twilio_client: Optional[TwilioRestClient] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ============================================================
# DB MODEL
# ============================================================

class EstimateSession(Base):
    __tablename__ = "estimate_sessions"

    id = Column(Integer, primary_key=True, index=True)
    shop_token = Column(String(100), index=True)
    phone = Column(String(50), index=True)
    analysis_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ============================================================
# SHOP CONFIG (from SHOPS_JSON env)
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
        raise RuntimeError("SHOPS_JSON env variable missing!")
    data = json.loads(raw)
    return {shop["webhook_token"]: ShopConfig(**shop) for shop in data}


shops = load_shops()

# ============================================================
# GOOGLE CALENDAR
# ============================================================

def get_calendar_service():
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var")

    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)

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


def save_analysis(shop_token: str, phone: str, analysis: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        rec = EstimateSession(
            shop_token=shop_token,
            phone=phone,
            analysis_json=json.dumps(analysis),
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()


def get_latest_analysis(shop_token: str, phone: str) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
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
    finally:
        db.close()


def send_sms(to_number: str, body: str) -> None:
    """Send SMS via Twilio REST. Fails silently if Twilio creds not set."""
    if not twilio_client or not TWILIO_FROM_NUMBER:
        print("Twilio client not configured; cannot send SMS.")
        return
    try:
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=to_number,
        )
    except Exception as e:
        print("Error sending SMS via Twilio:", e)

# ============================================================
# AI — DAMAGE ANALYSIS (Vision via URL)
# ============================================================

def analyze_damage(image_url: str, shop: ShopConfig) -> Dict[str, Any]:
    pricing = json.dumps(shop.pricing, indent=2)

    prompt = f"""
You are an elite collision damage estimator in Ontario, Canada.

Viewpoint rules:
- Always describe damage from the DRIVER'S point of view.
- "driver side" = left side, "passenger side" = right side.

Using this shop pricing:
{pricing}

Your job:
1. Identify all damaged areas and parts.
2. Describe damage types (scratches, scuffs, dents, deep dents, cracks, deformation, structural).
3. Classify overall severity: "minor", "moderate", or "severe".
4. Estimate realistic labor hours (body + paint).
5. Estimate cost range in CAD based on the pricing context.

Return ONLY valid JSON with keys:
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
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )

    raw = resp["choices"][0]["message"]["content"]
    return safe_json_parse(raw)

# ============================================================
# AI — BOOKING PARSER
# ============================================================

def parse_booking_details(user_text: str, last_analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    summary = last_analysis.get("damage_summary", "") if last_analysis else ""

    prompt = f"""
Convert this customer's message into structured booking details for the auto body shop.

Customer message:
\"\"\"{user_text}\"\"\"

Damage summary:
\"\"\"{summary}\"\"\"

Return ONLY JSON with keys:
- name (string)
- phone (string)
- email (string)
- datetime_iso (string, ISO 8601, America/Toronto local time)
- notes (string)
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You output only JSON, no explanation."},
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
        print("Booking parse error:", e)
        return None

# ============================================================
# MESSAGE TEXT BUILDERS (for outbound SMS)
# ============================================================

def format_damage_analysis_text(shop: ShopConfig, analysis: Dict[str, Any]) -> str:
    return (
        f"🔍 AI Damage Analysis — {shop.name}\n\n"
        f"{analysis['damage_summary']}\n\n"
        f"Areas: {', '.join(analysis.get('areas', []))}\n"
        f"Damage Types: {', '.join(analysis.get('damage_types', []))}\n"
        f"Severity: {analysis.get('severity', '').title()}\n\n"
        "If this looks correct, reply 1.\n"
        "If you’d like to upload more photos, reply 2."
    )


def format_full_estimate_text(shop: ShopConfig, analysis: Dict[str, Any]) -> str:
    estimate_id = str(uuid.uuid4())
    return (
        f"📘 Full Estimate Breakdown (ID: {estimate_id})\n\n"
        f"Estimated Cost Range: ${analysis['estimated_cost_min']} – ${analysis['estimated_cost_max']} CAD\n"
        f"Recommended Labor Hours: {analysis['recommended_labor_hours']} hours\n\n"
        "This is a visual, preliminary estimate and may change after in-person inspection.\n\n"
        "To book an appointment, reply in ONE message with your:\n"
        "- Full Name\n- Phone Number\n- Email\n- Preferred Date & Time"
    )

# ============================================================
# GOOGLE CALENDAR EVENT
# ============================================================

def create_calendar_event(shop: ShopConfig, booking: Dict[str, Any], analysis: Optional[Dict[str, Any]]) -> None:
    service = get_calendar_service()

    description_lines = [
        f"Name: {booking.get('name', '')}",
        f"Phone: {booking.get('phone', '')}",
        f"Email: {booking.get('email', '')}",
        "",
        "Customer Notes:",
        booking.get("notes", ""),
        "",
        "AI Visual Estimate:",
    ]

    if analysis:
        description_lines.append(analysis.get("damage_summary", ""))
        description_lines.append("")
        description_lines.append(f"Areas: {', '.join(analysis.get('areas', []))}")
        description_lines.append(f"Damage Types: {', '.join(analysis.get('damage_types', []))}")
        description_lines.append(f"Severity: {analysis.get('severity', '').title()}")
        description_lines.append(
            f"Est. Cost: ${analysis.get('estimated_cost_min')} – ${analysis.get('estimated_cost_max')} CAD"
        )
        description_lines.append(
            f"Recommended Labor Hours: {analysis.get('recommended_labor_hours')}"
        )

    description = "\n".join(description_lines)

    start_dt = booking["datetime_iso"]
    start = datetime.fromisoformat(start_dt)
    end = start + timedelta(hours=1)

    event = {
        "summary": f"AI Estimate Booking — {booking.get('name', 'Customer')}",
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/Toronto"},
    }

    service.events().insert(
        calendarId=shop.calendar_id,
        body=event
    ).execute()

# ============================================================
# BACKGROUND TASKS
# ============================================================

def bg_process_image(shop_token: str, shop: ShopConfig, from_phone: str, media_url: str):
    """
    Runs after webhook has already replied to Twilio.
    Does the slow OpenAI vision call + outbound SMS.
    """
    try:
        analysis = analyze_damage(media_url, shop)
        save_analysis(shop_token, from_phone, analysis)
        body = format_damage_analysis_text(shop, analysis)
        send_sms(from_phone, body)
        print("Background image analysis complete and SMS sent.")
    except Exception as e:
        print("Background image processing error:", e)
        send_sms(
            from_phone,
            "⚠️ There was an issue analyzing your photo. Please try another angle or a clearer image."
        )

def bg_process_booking(shop: ShopConfig, from_phone: str, user_text: str, last_analysis: Optional[Dict[str, Any]]):
    try:
        booking = parse_booking_details(user_text, last_analysis)
        if not booking:
            send_sms(from_phone, "⚠️ I couldn't read your booking details. Please resend with name, phone, email, and preferred date & time.")
            return
        create_calendar_event(shop, booking, last_analysis)
        send_sms(from_phone, "✅ Your appointment has been booked. The shop will contact you if any changes are needed.")
    except Exception as e:
        print("Background booking error:", e)
        send_sms(from_phone, "⚠️ There was a problem booking your appointment. Please try again or call the shop directly.")

# ============================================================
# MAIN TWILIO WEBHOOK — FAST RESPONSES ONLY
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio webhook: https://YOUR-APP/sms-webhook?token=shop_miss_123
    """
    form = await request.form()

    token = request.query_params.get("token")
    if not token or token not in shops:
        return PlainTextResponse("Invalid token", status_code=401)

    shop = shops[token]

    user_msg_raw: str = form.get("Body", "") or ""
    user_msg = user_msg_raw.strip()
    user_msg_lower = user_msg.lower()
    media_url = form.get("MediaUrl0")
    from_phone = form.get("From", "").strip() or "unknown"

    # 1) Greeting / first touch
    if not media_url and user_msg_lower in ("", "hi", "hello", "hey", "start"):
        msg = MessagingResponse()
        msg.message(
            f"👋 Welcome to {shop.name}!\n\n"
            "Please send 1–3 clear photos of the damage for a fast AI estimate."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    # 2) Image received → queue background vision job and reply instantly
    if media_url:
        # queue background analysis
        background_tasks.add_task(bg_process_image, token, shop, from_phone, media_url)

        msg = MessagingResponse()
        msg.message(
            "📸 Thanks! We received your photos.\n\n"
            "Our AI is analyzing the damage now — you'll get your estimate in a moment."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    # 3) User confirms analysis is correct → send full estimate immediately
    if user_msg_lower == "1":
        last_analysis = get_latest_analysis(token, from_phone)
        msg = MessagingResponse()
        if not last_analysis:
            msg.message("I couldn't find your last estimate. Please resend your photos.")
        else:
            msg.message(format_full_estimate_text(shop, last_analysis))
        return PlainTextResponse(str(msg), media_type="application/xml")

    # 4) User wants to upload more photos
    if user_msg_lower == "2":
        msg = MessagingResponse()
        msg.message("No problem! Please send a few more photos of the damage.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    # 5) Booking flow — queue to background (could also be heavy)
    last_analysis = get_latest_analysis(token, from_phone)
    background_tasks.add_task(bg_process_booking, shop, from_phone, user_msg_raw, last_analysis)

    msg = MessagingResponse()
    msg.message(
        "📅 Thanks! We're processing your booking request now.\n"
        "You'll receive a confirmation text shortly."
    )
    return PlainTextResponse(str(msg), media_type="application/xml")
