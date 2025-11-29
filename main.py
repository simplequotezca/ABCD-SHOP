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

# -------------------------------
# OPENAI (Stable Vision)
# -------------------------------
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI()

# ============================================================
# DATABASE SETUP
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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
# SHOP CONFIG
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
        raise RuntimeError("SHOPS_JSON environment variable missing!")
    arr = json.loads(raw)
    return {shop["webhook_token"]: ShopConfig(**shop) for shop in arr}


shops = load_shops()

# ============================================================
# GOOGLE CALENDAR SERVICE
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

def fetch_image_as_base64(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def safe_json_parse(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)

# ============================================================
# AI DAMAGE ANALYSIS — Vision Working Version
# ============================================================

def analyze_damage(image_base64: str, shop: ShopConfig) -> Dict[str, Any]:
    pricing_json = json.dumps(shop.pricing, indent=2)

    prompt = f"""
You are an elite collision estimator with 20+ years experience.
Analyze the damage from the DRIVER'S POINT OF VIEW only.
Use this shop pricing:
{pricing_json}

Return strict JSON:
- damage_summary (string)
- areas (string list)
- damage_types (string list)
- severity: minor | moderate | severe
- recommended_labor_hours: number
- estimated_cost_min: number
- estimated_cost_max: number
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You output only JSON. No extra text."
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)

# ============================================================
# BOOKING PARSER
# ============================================================

def parse_booking_info(user_text: str, analysis: Optional[Dict[str, Any]]):
    summary = analysis.get("damage_summary") if analysis else ""

    prompt = f"""
Convert the user's message into booking details.

User message:
\"\"\"{user_text}\"\"\"

Damage Summary:
\"\"\"{summary}\"\"\"

Return JSON:
- name
- phone
- email
- datetime_iso (ISO time, America/Toronto)
- notes
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)

# ============================================================
# DB HELPERS
# ============================================================

def save_analysis(shop_token: str, phone: str, analysis: Dict[str, Any]):
    db = SessionLocal()
    try:
        db.add(EstimateSession(
            shop_token=shop_token,
            phone=phone,
            analysis_json=json.dumps(analysis),
        ))
        db.commit()
    finally:
        db.close()


def load_latest_analysis(shop_token: str, phone: str) -> Optional[Dict[str, Any]]:
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
        return json.loads(rec.analysis_json) if rec else None
    finally:
        db.close()

# ============================================================
# MESSAGE BUILDERS
# ============================================================

def welcome_msg(shop: ShopConfig):
    r = MessagingResponse()
    r.message(
        f"👋 Welcome to *{shop.name}*! \n\n"
        "Please send 1–3 clear photos of the damage for an instant AI estimate."
    )
    return str(r)


def damage_msg(shop: ShopConfig, a: Dict[str, Any]):
    r = MessagingResponse()
    r.message(
        f"🔍 *AI Damage Analysis*\n"
        f"{a['damage_summary']}\n\n"
        f"Areas: {', '.join(a['areas'])}\n"
        f"Damage Types: {', '.join(a['damage_types'])}\n"
        f"Severity: *{a['severity'].title()}*\n\n"
        "If accurate, reply *1*.\n"
        "If you'd like to upload more photos, reply *2*."
    )
    return str(r)


def full_estimate_msg(shop: ShopConfig, a: Dict[str, Any]):
    r = MessagingResponse()
    r.message(
        f"📘 *Full Estimate*\n\n"
        f"Cost Range: ${a['estimated_cost_min']} – ${a['estimated_cost_max']}\n"
        f"Labor Hours: {a['recommended_labor_hours']}\n\n"
        "To book an appointment, reply (all in one message):\n"
        "- Name\n"
        "- Phone\n"
        "- Email\n"
        "- Preferred Date & Time"
    )
    return str(r)


def booked_msg():
    r = MessagingResponse()
    r.message("✅ Your appointment has been booked! The shop will see you soon.")
    return str(r)

# ============================================================
# GOOGLE CALENDAR BOOKING
# ============================================================

def create_calendar_event(shop: ShopConfig, details: Dict[str, Any], analysis: Dict[str, Any]):
    service = get_calendar_service()

    desc = f"""
Name: {details['name']}
Phone: {details['phone']}
Email: {details['email']}

Damage Summary:
{analysis['damage_summary']}

Areas: {', '.join(analysis['areas'])}
Damage Types: {', '.join(analysis['damage_types'])}
Severity: {analysis['severity']}
Cost Range: ${analysis['estimated_cost_min']} – ${analysis['estimated_cost_max']}
"""

    start = datetime.fromisoformat(details['datetime_iso'])
    end = start + timedelta(hours=1)

    event = {
        "summary": f"AI Estimate — {details['name']}",
        "description": desc,
        "start": {"dateTime": start.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/Toronto"},
    }

    service.events().insert(
        calendarId=shop.calendar_id,
        body=event
    ).execute()

# ============================================================
# MAIN TWILIO ENDPOINT
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()

    token = request.query_params.get("token")
    if not token or token not in shops:
        return PlainTextResponse("Invalid token", status_code=401)

    shop = shops[token]

    body = form.get("Body", "").strip()
    media_url = form.get("MediaUrl0")
    phone = form.get("From", "")

    # 1 — welcome
    if not media_url and body.lower() in ("", "hi", "hello", "hey"):
        return PlainTextResponse(welcome_msg(shop), media_type="application/xml")

    # 2 — image → analyze
    if media_url:
        try:
            img64 = fetch_image_as_base64(media_url)
            analysis = analyze_damage(img64, shop)
            save_analysis(token, phone, analysis)
            return PlainTextResponse(damage_msg(shop, analysis), media_type="application/xml")
        except Exception as e:
            err = MessagingResponse()
            err.message("⚠️ Could not analyze the image. Please send another angle.")
            return PlainTextResponse(str(err), media_type="application/xml")

    # 3 — confirm analysis
    if body == "1":
        a = load_latest_analysis(token, phone)
        if not a:
            msg = MessagingResponse()
            msg.message("Please resend the photos.")
            return PlainTextResponse(str(msg), media_type="application/xml")
        return PlainTextResponse(full_estimate_msg(shop, a), media_type="application/xml")

    # 4 — more photos
    if body == "2":
        msg = MessagingResponse()
        msg.message("Please send more photos.")
        return PlainTextResponse(str(msg), media_type="application/xml")

    # 5 — booking
    a = load_latest_analysis(token, phone)
    if a:
        try:
            booking = parse_booking_info(body, a)
            create_calendar_event(shop, booking, a)
            return PlainTextResponse(booked_msg(), media_type="application/xml")
        except Exception:
            msg = MessagingResponse()
            msg.message("⚠️ Could not process booking. Check formatting.")
            return PlainTextResponse(str(msg), media_type="application/xml")

    # 6 — fallback
    msg = MessagingResponse()
    msg.message("Please send vehicle photos to begin your estimate.")
    return PlainTextResponse(str(msg), media_type="application/xml")
