import os
import json
import uuid
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

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
# FASTAPI + OPENAI SETUP
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# DATABASE (POSTGRES / SQLALCHEMY)
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

# Fix old postgres:// format if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EstimateSession(Base):
    """
    Stores the latest AI analysis per (shop_token, phone).
    This lets us:
    - Show analysis when user sends photos
    - On '1', pull the saved analysis and send full estimate
    - On booking, attach the analysis into the calendar event
    """
    __tablename__ = "estimate_sessions"

    id = Column(Integer, primary_key=True, index=True)
    shop_token = Column(String(100), index=True)
    phone = Column(String(50), index=True)
    analysis_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


Base.metadata.create_all(bind=engine)

# ============================================================
# MULTI-SHOP CONFIG (FROM SHOPS_JSON ENV)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: Dict[str, Any]
    hours: Dict[str, Any]


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
          "monday": {"open": "09:00", "close": "17:00"}
        }
      }
    ]
    """
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON env variable missing!")
    data = json.loads(raw)
    return {shop["webhook_token"]: ShopConfig(**shop) for shop in data}


shops = load_shops()

# ============================================================
# GOOGLE CALENDAR SERVICE
# ============================================================

def get_calendar_service():
    """
    Requires GOOGLE_SERVICE_ACCOUNT_JSON env var.
    You must share each shop's calendar with the service account email.
    """
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var")

    info = json.loads(raw_json)

    creds = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=creds)
    return service

# ============================================================
# UTILS
# ============================================================

def fetch_image_as_base64(url: str) -> str:
    """
    Download image from Twilio and convert to base64.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def safe_json_parse(raw: str) -> Any:
    """
    Handle cases where the model wraps JSON in ```json ... ```
    """
    text = raw.strip()
    if text.startswith("```"):
        # strip first ```
        parts = text.split("```")
        # usually ['', 'json\n{...}', '']
        if len(parts) >= 2:
            text = parts[1]
        text = text.lstrip("json").strip()
    return json.loads(text)

# ============================================================
# AI — DAMAGE ANALYSIS (DRIVER POV, ULTRA ACCURATE)
# ============================================================

def analyze_damage(image_base64: str, shop: ShopConfig) -> Dict[str, Any]:
    """
    Calls OpenAI vision to analyze vehicle damage with pricing context.
    Must return a JSON dict with:
    - damage_summary (string)
    - areas (list of strings)
    - damage_types (list of strings)
    - severity ("minor" | "moderate" | "severe")
    - recommended_labor_hours (number)
    - estimated_cost_min (number)
    - estimated_cost_max (number)
    """
    pricing = json.dumps(shop.pricing, indent=2)

    prompt = f"""
You are an elite automotive collision damage estimator in Ontario, Canada.
You MUST analyze damage strictly from the driver's point of view:
- "driver side" = left side
- "passenger side" = right side

Use this pricing context:
{pricing}

Instructions:
1. Identify all damaged panels and parts (bumper, fender, hood, trunk, doors, lights, etc.).
2. Describe severity and type of damage (scratches, scuffs, dents, deep dents, cracks, deformation, structural).
3. Classify overall severity as one of: "minor", "moderate", "severe".
4. Estimate realistic labor hours, considering body and paint.
5. Estimate cost range in CAD using shop pricing and this severity baseline:
   - minor: 350–650
   - moderate: 900–1600
   - severe: 2000–5000+
6. Be conservative but realistic. This is a visual PRELIMINARY estimate.

Return ONLY valid JSON with:
- "damage_summary": string
- "areas": string[]
- "damage_types": string[]
- "severity": "minor" | "moderate" | "severe"
- "recommended_labor_hours": number
- "estimated_cost_min": number
- "estimated_cost_max": number
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional auto body collision estimator. You output strict JSON, no explanations."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            },
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)

# ============================================================
# AI — BOOKING PARSER (FREE TEXT → STRUCTURED INFO)
# ============================================================

def parse_booking_details(user_text: str, last_analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Take freeform text (name, phone, email, date/time in ANY order)
    and convert into structured JSON:
    {
      "name": "...",
      "phone": "...",
      "email": "...",
      "datetime_iso": "2025-11-30T14:30:00",
      "notes": "..."
    }
    Returns None if parse fails.
    """
    analysis_summary = ""
    if last_analysis:
        analysis_summary = last_analysis.get("damage_summary", "")

    prompt = f"""
You are converting a customer's message into structured booking info for an auto body shop.

Customer message:
\"\"\"{user_text}\"\"\"

Latest damage estimate summary:
\"\"\"{analysis_summary}\"\"\"

Rules:
- Guess reasonable formatting if needed.
- The datetime must be in ISO-8601 local time for America/Toronto (e.g. "2025-11-30T14:30:00").
- If you cannot find a field, put an empty string "" for that field.

Return ONLY valid JSON with keys:
- name (string)
- phone (string)
- email (string)
- datetime_iso (string, ISO-8601)
- notes (string)
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only JSON. No extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        data = safe_json_parse(raw)
        # basic validation
        if not data.get("datetime_iso"):
            return None
        return data
    except Exception:
        return None

# ============================================================
# DB HELPERS
# ============================================================

def save_analysis(shop_token: str, phone: str, analysis: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        record = EstimateSession(
            shop_token=shop_token,
            phone=phone,
            analysis_json=json.dumps(analysis),
        )
        db.add(record)
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

# ============================================================
# MESSAGE BUILDERS (TWILIO FLOW)
# ============================================================

def send_welcome(shop: ShopConfig) -> str:
    msg = MessagingResponse()
    msg.message(
        f"👋 Welcome to *{shop.name}!* \n\n"
        "To get a fast AI estimate, please send *1–3 clear photos* of the damage "
        "from a few angles (close and far)."
    )
    return str(msg)


def send_damage_analysis(shop: ShopConfig, analysis: Dict[str, Any]) -> str:
    msg = MessagingResponse()
    msg.message(
        f"🔍 *AI Damage Analysis*\n"
        f"{analysis['damage_summary']}\n\n"
        f"Areas: {', '.join(analysis.get('areas', []))}\n"
        f"Damage Types: {', '.join(analysis.get('damage_types', []))}\n"
        f"Severity: *{analysis.get('severity', '').title()}*\n\n"
        "If this looks correct, reply *1*.\n"
        "If you’d like to upload more photos, reply *2*."
    )
    return str(msg)


def send_full_estimate(shop: ShopConfig, analysis: Dict[str, Any]) -> str:
    estimate_id = str(uuid.uuid4())
    cost_min = analysis.get("estimated_cost_min")
    cost_max = analysis.get("estimated_cost_max")
    hours = analysis.get("recommended_labor_hours")

    msg = MessagingResponse()
    msg.message(
        f"📘 *Full Estimate Breakdown* (ID: {estimate_id})\n\n"
        f"Estimated Cost Range (visual estimate): "
        f"${cost_min} – ${cost_max} CAD\n"
        f"Recommended Labor Hours: ~{hours} hours\n\n"
        "This is a preliminary AI estimate based on the photos and may be adjusted "
        "after an in-person inspection.\n\n"
        "📅 To book an appointment, please reply in ONE message with your:\n"
        "- Full Name\n"
        "- Phone Number\n"
        "- Email\n"
        "- Preferred Date & Time\n\n"
        "You can send it in any order. We’ll confirm your spot and notify the shop instantly."
    )
    return str(msg)


def confirm_booking() -> str:
    msg = MessagingResponse()
    msg.message(
        "✅ Your appointment request has been received and added to our schedule.\n\n"
        "A team member from the shop will reach out if any adjustments are needed. "
        "Thank you!"
    )
    return str(msg)


def ask_for_more_photos() -> str:
    msg = MessagingResponse()
    msg.message("No problem at all! 📸\nPlease upload a few more clear photos of the damage.")
    return str(msg)


def generic_confusion() -> str:
    msg = MessagingResponse()
    msg.message(
        "Sorry, I didn’t quite catch that.\n\n"
        "👉 To continue:\n"
        "- Send *photos* of the damage for an AI estimate, or\n"
        "- If you've already received an estimate, reply *1* to see the full breakdown, or\n"
        "- Reply with your *name, phone, email, and preferred date/time* to book."
    )
    return str(msg)

# ============================================================
# CALENDAR EVENT CREATION
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
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": "America/Toronto",
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": "America/Toronto",
        },
    }

    service.events().insert(
        calendarId=shop.calendar_id,
        body=event
    ).execute()

# ============================================================
# MAIN TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio webhook URL pattern:
    https://your-app-url/sms-webhook?token=shop_miss_123
    """
    form = await request.form()

    # token is passed as query parameter in the Twilio webhook URL
    token = request.query_params.get("token")
    if not token or token not in shops:
        return PlainTextResponse("Invalid token", status_code=401)

    shop = shops[token]

    user_msg_raw: str = form.get("Body", "") or ""
    user_msg = user_msg_raw.strip()
    user_msg_lower = user_msg.lower()
    media_url = form.get("MediaUrl0")
    from_phone = form.get("From", "").strip() or "unknown"

    # 1) First touch / greeting
    if not media_url and user_msg_lower in ("", "hi", "hello", "hey", "start"):
        twiml = send_welcome(shop)
        return PlainTextResponse(twiml, media_type="application/xml")

    # 2) User sends image(s) → run AI damage analysis
    if media_url:
        try:
            img64 = fetch_image_as_base64(media_url)
            analysis = analyze_damage(img64, shop)
            save_analysis(token, from_phone, analysis)
            twiml = send_damage_analysis(shop, analysis)
            return PlainTextResponse(twiml, media_type="application/xml")
        except Exception as e:
            msg = MessagingResponse()
            msg.message(
                "⚠️ There was an issue analyzing the image. "
                "Please try sending a clearer photo or a different angle."
            )
            return PlainTextResponse(str(msg), media_type="application/xml")

    # 3) User confirms analysis is correct → send full estimate
    if user_msg_lower == "1":
        last_analysis = get_latest_analysis(token, from_phone)
        if not last_analysis:
            msg = MessagingResponse()
            msg.message(
                "I couldn’t find the last estimate in the system. "
                "Please resend a photo so I can re-analyze the damage."
            )
            return PlainTextResponse(str(msg), media_type="application/xml")

        twiml = send_full_estimate(shop, last_analysis)
        return PlainTextResponse(twiml, media_type="application/xml")

    # 4) User wants to upload more photos
    if user_msg_lower == "2":
        twiml = ask_for_more_photos()
        return PlainTextResponse(twiml, media_type="application/xml")

    # 5) Assume they’re trying to book (free-form message with details)
    #    Use AI to parse booking info + attach last analysis into calendar.
    last_analysis = get_latest_analysis(token, from_phone)
    booking_data = parse_booking_details(user_msg_raw, last_analysis)

    if booking_data:
        try:
            create_calendar_event(shop, booking_data, last_analysis)
            twiml = confirm_booking()
            return PlainTextResponse(twiml, media_type="application/xml")
        except Exception as e:
            msg = MessagingResponse()
            msg.message(
                "⚠️ I had trouble booking that appointment. "
                "Please double-check your date/time format or try again."
            )
            return PlainTextResponse(str(msg), media_type="application/xml")

    # 6) Fallback if we couldn’t understand message
    twiml = generic_confusion()
    return PlainTextResponse(twiml, media_type="application/xml")
