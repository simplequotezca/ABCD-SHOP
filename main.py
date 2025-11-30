import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import requests
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioRestClient

from openai import OpenAI  # OpenAI client

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
client: Optional[OpenAI] = None

if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
    except Exception as e:
        print("ERROR initializing OpenAI client:", e)
else:
    print("WARNING: OPENAI_API_KEY missing – AI features disabled.")

# ============================================================
# DB INIT
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
# MULTI-SHOP CONFIG
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
# GOOGLE CALENDAR
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
# TWILIO OUTBOUND
# ============================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

twilio_client: Optional[TwilioRestClient] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("Twilio client initialized.")
    except Exception as e:
        print("ERROR initializing Twilio client:", e)
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
    try:
        return SessionLocal()
    except Exception as e:
        print("ERROR opening DB session:", e)
        return None


def save_analysis(shop_token: str, phone: str, analysis: Dict[str, Any]) -> None:
    db = get_db_session()
    if not db:
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
    if not db:
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
        return json.loads(rec.analysis_json) if rec else None
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
# AI DAMAGE ANALYSIS
# ============================================================

def can_use_ai() -> bool:
    return client is not None


def analyze_damage(image_url: str, shop: ShopConfig) -> Dict[str, Any]:
    if not can_use_ai():
        raise RuntimeError("AI not configured")

    pricing = json.dumps(shop.pricing, indent=2)

    system_prompt = (
        "You are an elite auto body collision estimator in Ontario, Canada."
    )

    user_prompt = f"""
Return a single JSON object describing detailed collision damage.
Pricing:
{pricing}
"""

    # FINAL CORRECT GPT-4.1 VISION IMAGE FORMAT
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},

                    # ---------------- CORRECT FORMAT ----------------
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                    # ------------------------------------------------
                ],
            },
        ],
        temperature=0.1,
        max_tokens=900,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)

# ============================================================
# BOOKING PARSER
# ============================================================

def parse_booking(user_text: str, analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not can_use_ai():
        return None

    summary = analysis.get("damage_summary", "") if analysis else ""

    system_prompt = (
        "You convert messages into booking JSON. Output ONLY valid JSON."
    )

    user_prompt = f"""
Convert to booking JSON:
Message:
{user_text}

Damage summary:
{summary}

Required keys:
- name
- phone
- email
- datetime_iso
- notes
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content
        data = safe_json_parse(raw)
        if not data.get("datetime_iso"):
            return None
        return data
    except Exception as e:
        print("ERROR in parse_booking:", e)
        return None

# ============================================================
# TEXT BUILDERS (UNCHANGED)
# ============================================================

def _format_parts_required(parts: List[Dict[str, Any]]) -> str:
    if not parts:
        return "None noted."
    lines = []
    for p in parts:
        part_name = p.get("part", "Part")
        op = p.get("operation", "inspect")
        lines.append(f"- {part_name} ({op})")
    return "\n".join(lines)


def _format_list_labelled(label: str, items: List[str]) -> str:
    if not items:
        return f"{label}: None noted."
    return f"{label}: " + ", ".join(items)


def format_damage_text(shop: ShopConfig, a: Dict[str, Any]) -> str:
    damage_summary = a.get("damage_summary", "Damage summary not available.")
    areas = a.get("areas", []) or []
    damage_types = a.get("damage_types", []) or []
    severity = (a.get("severity") or "").title() or "Unknown"

    parts_required = a.get("parts_required", []) or []
    structural_concerns = a.get("structural_concerns", []) or []
    paint_operations = a.get("paint_operations", []) or []
    recommended_actions = a.get("recommended_actions", []) or []
    notes = a.get("notes", "")
    confidence = a.get("confidence_score", None)

    parts_block = _format_parts_required(parts_required)
    structural_block = _format_list_labelled("Structural concerns to inspect", structural_concerns)
    paint_block = _format_list_labelled("Paint operations", paint_operations)

    actions_block = "Recommended actions: "
    if recommended_actions:
        actions_block += "; ".join(recommended_actions)
    else:
        actions_block += "Standard repair and refinish as needed."

    confidence_line = ""
    if confidence is not None:
        try:
            pct = int(float(confidence) * 100)
            confidence_line = f"\nConfidence: ~{pct}% (visual-only estimate)"
        except Exception:
            pass

    return (
        f"🔍 AI Damage Analysis — {shop.name}\n\n"
        f"{damage_summary}\n\n"
        f"Areas affected: {', '.join(areas) if areas else 'Not clearly detected.'}\n"
        f"Damage types: {', '.join(damage_types) if damage_types else 'Not clearly detected.'}\n"
        f"Severity: {severity}\n\n"
        f"Parts & operations:\n{parts_block}\n\n"
        f"{structural_block}\n"
        f"{paint_block}\n"
        f"{actions_block}"
        f"{confidence_line}\n\n"
        "If this looks correct, reply 1.\n"
        "If you’d like to upload more photos, reply 2."
    )


def format_full_estimate_text(shop: ShopConfig, a: Dict[str, Any]) -> str:
    eid = str(uuid.uuid4())

    cost_min = a.get("estimated_cost_min", 0)
    cost_max = a.get("estimated_cost_max", 0)
    hours = a.get("recommended_labor_hours", 0)

    structural_concerns = a.get("structural_concerns", []) or []
    paint_operations = a.get("paint_operations", []) or []

    structural_line = (
        "Structural concerns: " + (", ".join(structural_concerns) if structural_concerns else "None noted.")
    )
    paint_line = (
        "Paint operations: " + (", ".join(paint_operations) if paint_operations else "Standard prep, basecoat, clearcoat.")
    )

    return (
        f"📘 Full AI Estimate — {shop.name}\n"
        f"Estimate ID: {eid}\n\n"
        f"Estimated cost range: ${cost_min} – ${cost_max} CAD\n"
        f"Estimated labor: {hours} hours\n\n"
        f"{structural_line}\n{paint_line}\n\n"
        "This is a visual preliminary estimate. Final pricing may change after an in-person inspection.\n\n"
        "To book an appointment, reply with your name, phone number, email, and preferred date/time."
    )

# ============================================================
# CALENDAR
# ============================================================

def create_calendar_event(shop: ShopConfig, booking: Dict[str, Any], analysis: Optional[Dict[str, Any]]) -> None:
    service = get_calendar_service()
    if not service:
        print("Calendar not available.")
        return

    lines = [
        f"Name: {booking.get('name', '')}",
        f"Phone: {booking.get('phone', '')}",
        f"Email: {booking.get('email', '')}",
        "",
        "Notes:",
        booking.get("notes", ""),
    ]

    if analysis:
        lines.extend([
            "",
            "AI Estimate Summary:",
            analysis.get("damage_summary", ""),
            f"Areas: {', '.join(analysis.get('areas', []))}",
            f"Types: {', '.join(analysis.get('damage_types', []))}",
            f"Severity: {analysis.get('severity', '')}",
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
            send_sms(from_phone, "Our AI is temporarily unavailable.")
            return

        analysis = analyze_damage(media_url, shop)
        save_analysis(shop_token, from_phone, analysis)
        send_sms(from_phone, format_damage_text(shop, analysis))

    except Exception as e:
        print("Background image error:", e)
        send_sms(
            from_phone,
            "⚠️ Issue analyzing your photo. Please try another angle."
        )


def bg_process_booking(shop: ShopConfig, from_phone: str, text: str, last_analysis: Optional[Dict[str, Any]]):
    try:
        booking = parse_booking(text, last_analysis)
        if not booking:
            send_sms(
                from_phone,
                "⚠️ I couldn't understand your booking. Please resend your name, phone, email, and preferred date/time."
            )
            return

        create_calendar_event(shop, booking, last_analysis)
        send_sms(
            from_phone,
            "✅ Your appointment request has been added to the shop calendar."
        )

    except Exception as e:
        print("Background booking error:", e)
        send_sms(
            from_phone,
            "⚠️ Error booking your appointment. Please try again."
        )

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
        msg.message("This shop is not configured.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    body_raw: str = form.get("Body", "") or ""
    lower = body_raw.strip().lower()
    media_url = form.get("MediaUrl0")
    from_phone = form.get("From", "").strip() or "unknown"

    if not media_url and lower in ("", "hi", "hello", "hey", "start"):
        msg = MessagingResponse()
        msg.message(
            f"👋 Welcome to {shop.name}!\n\n"
            "Please send 1–3 clear photos of the damage for an AI estimate."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    if media_url:
        background_tasks.add_task(bg_process_image, token, shop, from_phone, media_url)
        msg = MessagingResponse()
        msg.message(
            "📸 Thanks! We received your photos.\n\n"
            "Our AI is analyzing the damage now – you'll get your detailed estimate shortly."
        )
        return PlainTextResponse(str(msg), media_type="application/xml")

    if lower == "1":
        last = get_latest_analysis(token, from_phone)
        msg = MessagingResponse()
        if not last:
            msg.message("I couldn't find your last estimate. Please resend your photos.")
        else:
            msg.message(format_full_estimate_text(shop, last))
        return PlainTextResponse(str(msg), media_type="application/xml")

    if lower == "2":
        msg = MessagingResponse()
        msg.message("No problem! Please send a few more photos.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    last = get_latest_analysis(token, from_phone)
    background_tasks.add_task(bg_process_booking, shop, from_phone, body_raw, last)

    msg = MessagingResponse()
    msg.message(
        "📅 Thanks! We're processing your booking details now."
    )
    return PlainTextResponse(str(msg), media_type="application/xml")
