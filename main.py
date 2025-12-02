import os
import json
import base64
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

import httpx
from dateutil import parser as dateparser

from openai import OpenAI

from googleapiclient.discovery import build
from google.oauth2 import service_account


# ============================================================
# BASIC SETUP
# ============================================================

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

SHOPS_JSON = os.getenv("SHOPS_JSON")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")


# ============================================================
# SHOP CONFIG
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None


SHOPS_BY_TOKEN: Dict[str, Shop] = {}


def load_shops():
    global SHOPS_BY_TOKEN
    try:
        shops = [Shop(**s) for s in json.loads(SHOPS_JSON)]
        SHOPS_BY_TOKEN = {s.webhook_token: s for s in shops}
        logger.info(f"Loaded {len(shops)} shops")
    except Exception as e:
        logger.error(f"Error loading shops: {e}")
        SHOPS_BY_TOKEN = {}


load_shops()


# ============================================================
# GOOGLE CALENDAR
# ============================================================

calendar_service = None
if GOOGLE_CREDENTIALS:
    try:
        creds_info = json.loads(GOOGLE_CREDENTIALS)
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        logger.info("Calendar ready")
    except Exception as e:
        logger.error(f"Google Calendar Error: {e}")


# ============================================================
# MEMORY STORE FOR LEADS + BOOKINGS
# ============================================================

LAST_ESTIMATES: Dict[str, Dict[str, Any]] = {}
VIN_CACHE: Dict[str, Dict[str, Any]] = {}


# ============================================================
# AI PROMPT
# ============================================================

AI_PROMPT = """
You are an Ontario collision estimator (2025 pricing). Output fields:
Severity, Estimate range, Areas damaged, Damage types, and a 2–4 sentence explanation.
"""


# ============================================================
# IMAGE DOWNLOAD
# ============================================================

async def download_image(media_url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as c:
            resp = await c.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
        if resp.status_code != 200:
            return None

        mime = resp.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    except Exception as e:
        logger.error(f"Image download error: {e}")
        return None


# ============================================================
# AI ESTIMATION
# ============================================================

async def ai_estimate(data_url: str, shop: Shop) -> str:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_PROMPT},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": f"Shop name: {shop.name}"},
                     {"type": "image_url", "image_url": data_url}
                 ]}
            ]
        )
        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return "⚠️ Error analyzing image. Please resend."


# ============================================================
# VIN DETECTION + DECODING (C6 Upgrade)
# ============================================================

def extract_vin(text: str) -> Optional[str]:
    vin = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text.upper())
    return vin[0] if vin else None


async def decode_vin(vin: str) -> Dict[str, Any]:
    if vin in VIN_CACHE:
        return VIN_CACHE[vin]

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvaluesextended/{vin}?format=json"

    try:
        async with httpx.AsyncClient(timeout=20) as c:
            resp = await c.get(url)
        data = resp.json()

        if "Results" not in data:
            return {}

        result = data["Results"][0]

        decoded = {
            "year": result.get("ModelYear"),
            "make": result.get("Make"),
            "model": result.get("Model"),
            "trim": result.get("Trim"),
            "body_class": result.get("BodyClass"),
            "engine": result.get("EngineModel"),
            "engine_size": result.get("DisplacementL"),
            "transmission": result.get("TransmissionStyle"),
            "drive": result.get("DriveType"),
            "manufacturer": result.get("Manufacturer"),
            "full_raw": result
        }

        VIN_CACHE[vin] = decoded
        return decoded

    except Exception as e:
        logger.error(f"VIN decode error: {e}")
        return {}


# ============================================================
# FLEXIBLE BOOKING PARSER (C5)
# ============================================================

def extract_time(text: str) -> Optional[str]:
    text = text.lower().replace(".", "")

    m = re.search(r"\b(\d{1,2}(:\d{2})?\s?(am|pm))\b", text)
    if m:
        return m.group(1)

    m = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", text)
    if m:
        return m.group(1)

    if "noon" in text:
        return "12:00pm"
    if "morning" in text:
        return "10:00am"
    if "evening" in text:
        return "6:00pm"

    return None


def extract_date(text: str) -> Optional[datetime]:
    text = text.lower()

    if "tomorrow" in text:
        return datetime.now() + timedelta(days=1)

    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }
    for word, idx in days_map.items():
        if word in text:
            today = datetime.now().weekday()
            delta = (idx - today) % 7
            delta = 7 if delta == 0 else delta
            return datetime.now() + timedelta(days=delta)

    try:
        return dateparser.parse(text, fuzzy=True)
    except:
        return None


def parse_booking_flex(body: str) -> Dict[str, str]:
    out = {"name": "", "email": "", "vehicle": "", "time": "", "date": "", "vin": ""}
    lower = body.lower()

    out["vin"] = extract_vin(body) or ""

    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", body)
    if email_match:
        out["email"] = email_match.group(0)

    t = extract_time(body)
    if t:
        out["time"] = t

    d = extract_date(body)
    if d:
        out["date"] = d.isoformat()

    words = re.findall(r"[A-Za-z]+", body)
    if words:
        out["name"] = " ".join(words[:2])

    remaining = body
    for v in [out["name"], out["email"], out["vin"]]:
        if v:
            remaining = remaining.replace(v, "")

    out["vehicle"] = remaining.strip()

    return out


# ============================================================
# CREATE EVENTS
# ============================================================

def create_lead_event(shop: Shop, phone: str, estimate: str):
    if not calendar_service or not shop.calendar_id:
        return

    now = datetime.utcnow()
    event = {
        "summary": f"AI Lead - {phone}",
        "start": {"dateTime": now.isoformat() + "Z"},
        "end": {"dateTime": (now + timedelta(hours=1)).isoformat() + "Z"},
        "description": f"Phone: {phone}\n\nAI Estimate:\n{estimate}",
    }

    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()
    except Exception as e:
        logger.error(f"Lead event error: {e}")


def create_booking_event(shop: Shop, info: Dict[str, str], estimate: str, phone: str, vin_data: Dict[str, Any]):
    if not calendar_service or not shop.calendar_id:
        return

    if info["date"]:
        dt = datetime.fromisoformat(info["date"])
    else:
        dt = datetime.now() + timedelta(days=1)

    start = dt
    end = dt + timedelta(hours=1)

    description = (
        f"Name: {info['name']}\n"
        f"Phone: {phone}\n"
        f"Email: {info['email']}\n"
        f"Vehicle: {info['vehicle']}\n"
        f"Time: {info['time']}\n"
        f"VIN: {info['vin']}\n\n"
        "Decoded Vehicle Info:\n"
        f"Year: {vin_data.get('year')}\n"
        f"Make: {vin_data.get('make')}\n"
        f"Model: {vin_data.get('model')}\n"
        f"Trim: {vin_data.get('trim')}\n"
        f"Engine: {vin_data.get('engine')}\n"
        f"Body: {vin_data.get('body_class')}\n"
        f"Drive: {vin_data.get('drive')}\n"
        f"Transmission: {vin_data.get('transmission')}\n\n"
        f"AI Estimate:\n{estimate}"
    )

    event = {
        "summary": f"AI Booking - {info['name']}",
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
        "description": description,
    }

    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()
    except Exception as e:
        logger.error(f"Booking event error: {e}")


# ============================================================
# SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404)

    form = await request.form()
    phone = form.get("From")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia") or "0")

    resp = MessagingResponse()

    # BOOKING
    if num_media == 0 and body.lower().startswith("book"):
        info = parse_booking_flex(body)

        vin_data = {}
        if info["vin"]:
            vin_data = await decode_vin(info["vin"])

        estimate = LAST_ESTIMATES.get(phone, {}).get("estimate", "(no estimate)")
        create_booking_event(shop, info, estimate, phone, vin_data)

        resp.message("✅ Appointment request received! The shop will confirm shortly.")
        return Response(str(resp), media_type="application/xml")

    # WELCOME MESSAGE
    if num_media == 0:
        resp.message(
            f"Welcome to {shop.name}!\n\n"
            "Send 1–3 photos for an AI estimate.\n\n"
            "To book later, reply: BOOK + any details (name, date, time, vehicle, VIN)."
        )
        return Response(str(resp), media_type="application/xml")

    # PHOTO -> AI ESTIMATE
    media_url = form.get("MediaUrl0")
    data_url = await download_image(media_url)

    if not data_url:
        resp.message("⚠️ Couldn't read the image. Please resend.")
        return Response(str(resp), media_type="application/xml")

    estimate = await ai_estimate(data_url, shop)

    LAST_ESTIMATES[phone] = {
        "estimate": estimate,
        "timestamp": datetime.now().isoformat()
    }

    create_lead_event(shop, phone, estimate)

    resp.message(
        estimate + "\n\n"
        "To book an appointment, reply: BOOK + your info (any order)."
    )
    return Response(str(resp), media_type="application/xml")


@app.get("/")
def root():
    return {"status": "running"}
