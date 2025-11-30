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
# TWILIO OUTBOUND CLIENT
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
# AI HELPERS
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

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},

                    # ------------------------------------------------------
                    # 🔧 FIX APPLIED HERE:
                    # Old (broken):
                    # {"type": "image_url", "image_url": {"url": image_url}}
                    # New (correct):
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                    # ------------------------------------------------------
                ],
            },
        ],
        temperature=0.1,
        max_tokens=900,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)

# ============================================================
# BOOKING PARSER, TEXT BUILDERS, CALENDAR, BACKGROUND TASKS
# (UNCHANGED — FULL CODE REMAINS EXACTLY AS YOUR ORIGINAL)
# ============================================================

# 🚨 **IMPORTANT**
# Because you asked for “ONLY the image block fix”, I am not repeating
# the remaining 700+ lines. They remain EXACTLY the same as your original
# file and do not need modification.

# 🚀 Just paste the fixed version of analyze_damage() above into your file
# and everything else stays exactly the same.
