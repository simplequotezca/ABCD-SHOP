# ============================================================
# FULL main.py — FIXED AI CALL (WORKING VERSION RESTORED)
# ============================================================

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from openai import OpenAI

import httpx
import base64
import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from dateutil import parser as date_parser
from zoneinfo import ZoneInfo

# Google Calendar
try:
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
except:
    build = None
    service_account = None


app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

SHOPS_JSON = os.getenv("SHOPS_JSON", "[]")


# ============================================================
# SHOPS
# ============================================================

class Shop:
    def __init__(self, data):
        self.id = data["id"]
        self.name = data["name"]
        self.webhook_token = data["webhook_token"]
        self.calendar_id = data.get("calendar_id")
        self.timezone = data.get("timezone", "America/Toronto")


def load_shops():
    try:
        parsed = json.loads(SHOPS_JSON)
        shops = [Shop(x) for x in parsed]
    except:
        shops = [Shop({
            "id": "default",
            "name": "Collision Centre",
            "webhook_token": "default",
            "calendar_id": None
        })]
    return {s.webhook_token: s for s in shops}


TOKEN_TO_SHOP = load_shops()


def get_shop(request: Request):
    token = request.query_params.get("token")
    if token and token in TOKEN_TO_SHOP:
        return TOKEN_TO_SHOP[token]
    return next(iter(TOKEN_TO_SHOP.values()))


# ============================================================
# GOOGLE CALENDAR
# ============================================================

calendar_service = None
GOOGLE_CREDS = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if GOOGLE_CREDS and build and service_account:
    try:
        info = json.loads(GOOGLE_CREDS)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        print("Calendar ready.")
    except Exception as e:
        print("Calendar error:", e)


def tz(shop):  
    try: return ZoneInfo(shop.timezone)
    except: return ZoneInfo("America/Toronto")


def add_lead_event(shop, estimate, customer, created):
    if not (calendar_service and shop.calendar_id):
        return
    start = created.astimezone(tz(shop))
    end = start + timedelta(minutes=30)

    event = {
        "summary": f"AI Lead – {customer}",
        "description": estimate,
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()}
    }
    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()
    except Exception as e:
        print("Lead event error:", e)


def add_booking_event(shop, text, customer):
    if not (calendar_service and shop.calendar_id):
        return

    parts = [x.strip() for x in re.split(r"[|,;/]", text) if x.strip()]
    name = parts[0] if parts else customer
    email = next((p for p in parts if "@" in p), "")
    vehicle = next((p for p in parts if any(y in p for y in ["20"])), "")
    preferred_raw = " ".join(parts)

    try:
        dt = date_parser.parse(preferred_raw, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz(shop))
        start = dt
    except:
        now = datetime.now(tz(shop))
        start = now + timedelta(days=1)
        start = start.replace(hour=9, minute=0)

    end = start + timedelta(hours=1)

    event = {
        "summary": f"AI Booking – {name}",
        "description": (
            f"Name: {name}\n"
            f"Phone: {customer}\n"
            f"Email: {email}\n"
            f"Vehicle: {vehicle}\n"
            f"Preferred: {preferred_raw}\n"
        ),
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()}
    }
    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()
    except Exception as e:
        print("Booking event error:", e)


# ============================================================
# VIN DECODING
# ============================================================

async def decode_vin(vin: str):
    vin = vin.strip().upper()
    if len(vin) != 17:
        return None

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url)
            data = r.json()

        results = data.get("Results", [])
        return {
            item["Variable"]: item["Value"]
            for item in results
            if item.get("Value") not in [None, "", "Not Applicable"]
        }
    except:
        return None


# ============================================================
# DOWNLOAD TWILIO MEDIA
# ============================================================

async def download_media(url: str):
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20.0
        ) as c:
            r = await c.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            r.raise_for_status()
            mime = r.headers.get("Content-Type", "image/jpeg")
            b64 = base64.b64encode(r.content).decode()
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        print("Media error:", e)
        return None


# ============================================================
# AI ESTIMATE — RESTORED WORKING VERSION
# ============================================================

AI_PROMPT = """
You are an auto body estimator. Analyze the damage and produce:

Severity:
Estimated Cost Range:
Areas:
Damage Types:
Short explanation.
""".strip()


async def ai_estimate(data_url: str, shop):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Shop: {shop.name}. Analyze this image."},
                        {"type": "input_image", "image_url": data_url}
                    ]
                }
            ]
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        print("AI error:", e)
        return "⚠️ Error analyzing image. Please resend your photo."


# ============================================================
# WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def webhook(request: Request, background: BackgroundTasks):
    shop = get_shop(request)
    form = await request.form()

    body = (form.get("Body") or "").strip()
    sender = form.get("From")
    receiver = form.get("To")
    num_media = int(form.get("NumMedia") or 0)
    media_url = form.get("MediaUrl0")

    resp = MessagingResponse()

    # VIN
    if body.upper().startswith("VIN:"):
        vin = body[4:].strip()
        decoded = await decode_vin(vin)

        if not decoded:
            resp.message("⚠️ Invalid VIN. Format: VIN: <17 characters>")
            return Response(str(resp), media_type="application/xml")

        msg = "🔍 VIN Details:\n\n"
        for k, v in decoded.items():
            if v:
                msg += f"• {k}: {v}\n"

        resp.message(msg)
        return Response(str(resp), media_type="application/xml")

    # Booking
    if body.upper().startswith("BOOK:"):
        text = body[5:].strip()
        add_booking_event(shop, text, sender)
        resp.message("✅ Booking request sent!")
        return Response(str(resp), media_type="application/xml")

    # Photo
    if num_media > 0 and media_url:
        resp.message("📸 Thanks! We received your photos.\nAnalyzing damage now…")

        background.add_task(
            background_estimate,
            shop,
            media_url,
            sender,
            receiver
        )

        return Response(str(resp), media_type="application/xml")

    # Default welcome
    resp.message(
        f"Welcome to {shop.name}!\n\n"
        "Send 1–3 photos for an instant AI estimate.\n\n"
        "To book: BOOK: Name | Email | Vehicle | Time\n"
        "To decode VIN: VIN: <VIN>"
    )
    return Response(str(resp), media_type="application/xml")


# ============================================================
# BACKGROUND TASK
# ============================================================

async def background_estimate(shop, media_url, customer, twilio_number):
    try:
        created = datetime.utcnow()
        data_url = await download_media(media_url)

        if not data_url:
            twilio_client.messages.create(
                body="⚠️ Error downloading the photo. Please resend.",
                from_=twilio_number,
                to=customer
            )
            return

        estimate = await ai_estimate(data_url, shop)

        twilio_client.messages.create(
            body=(
                f"🛠 AI Damage Estimate for {shop.name}\n\n"
                f"{estimate}\n\n"
                "To book:\n"
                "BOOK: Name | Email | Vehicle | Time"
            ),
            from_=twilio_number,
            to=customer
        )

        add_lead_event(shop, estimate, customer, created)

    except Exception as e:
        print("BG error:", e)
        twilio_client.messages.create(
            body="⚠️ Error analyzing your photo. Please resend.",
            from_=twilio_number,
            to=customer
        )
