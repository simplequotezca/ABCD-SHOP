# ============================================================
# FULL main.py — AI Estimator + Calendar + Booking + Leads + VIN
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
except Exception:
    build = None
    service_account = None


# ============================================================
# ENVIRONMENT
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

SHOPS_JSON = os.getenv("SHOPS_JSON", "[]")


# ============================================================
# SHOP STRUCTURE
# ============================================================

class Shop:
    def __init__(self, data: Dict):
        self.id = data["id"]
        self.name = data["name"]
        self.webhook_token = data["webhook_token"]
        self.calendar_id = data.get("calendar_id")
        self.timezone = data.get("timezone", "America/Toronto")


def load_shops():
    try:
        raw = json.loads(SHOPS_JSON)
        shops = [Shop(item) for item in raw]
    except:
        shops = [
            Shop({
                "id": "default",
                "name": "Collision Centre",
                "webhook_token": "default_token",
                "calendar_id": None
            })
        ]
    return {shop.webhook_token: shop for shop in shops}


TOKEN_TO_SHOP: Dict[str, Shop] = load_shops()


def get_shop_for_request(request: Request) -> Shop:
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
            info,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        calendar_service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        print("Calendar ready.")
    except Exception as e:
        print("Calendar init error:", e)


def tz_for(shop: Shop):
    try:
        return ZoneInfo(shop.timezone)
    except:
        return ZoneInfo("America/Toronto")


def add_lead_event(shop: Shop, estimate: str, customer: str, created_at: datetime):
    if not (calendar_service and shop.calendar_id):
        return

    tz = tz_for(shop)
    start = created_at.astimezone(tz)
    end = start + timedelta(minutes=30)

    event = {
        "summary": f"AI Lead – {customer}",
        "description": f"Customer: {customer}\n\n{estimate}",
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }

    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id,
            body=event
        ).execute()
    except Exception as e:
        print("Lead event error:", e)


def add_booking_event(shop: Shop, booking_text: str, customer: str):
    if not (calendar_service and shop.calendar_id):
        return

    # Parse “BOOK: Name | email | Vehicle | time”
    parts = [p.strip() for p in re.split(r"[|,;/]", booking_text) if p.strip()]

    name = parts[0] if len(parts) > 0 else customer
    email = ""
    vehicle = ""
    preferred = ""

    for p in parts:
        if "@" in p:
            email = p
        elif any(y in p for y in ["202", "201", "200"]):
            vehicle = p
        else:
            preferred += p + " "

    tz = tz_for(shop)
    try:
        parsed = date_parser.parse(preferred, fuzzy=True)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        start = parsed
    except:
        start = datetime.now(tz) + timedelta(days=1)
        start = start.replace(hour=9, minute=0)

    end = start + timedelta(hours=1)

    event = {
        "summary": f"AI Booking – {name}",
        "description": (
            f"Name: {name}\n"
            f"Phone: {customer}\n"
            f"Email: {email or 'N/A'}\n"
            f"Vehicle: {vehicle or 'N/A'}\n"
            f"Preferred: {preferred}\n"
            f"Raw text:\n{booking_text}"
        ),
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }

    if email:
        event["attendees"] = [{"email": email}]

    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()
    except Exception as e:
        print("Booking event error:", e)


# ============================================================
# VIN DECODER
# ============================================================

async def decode_vin(vin: str):
    vin = vin.strip().upper()
    if len(vin) != 17:
        return None

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url)
            r.raise_for_status()
            data = r.json()

        results = data.get("Results", [])
        decoded = {
            item["Variable"]: item["Value"]
            for item in results
            if item.get("Value") not in [None, "", "Not Applicable"]
        }
        return decoded or None
    except Exception as e:
        print("VIN error:", e)
        return None


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

async def download_media_as_data_url(url: str):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as c:
            r = await c.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            r.raise_for_status()
            mime = r.headers.get("Content-Type", "image/jpeg")
            b64 = base64.b64encode(r.content).decode()
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        print("Media download error:", e)
        return None


# ============================================================
# AI ESTIMATE
# ============================================================

AI_PROMPT = """
You are an expert auto body damage estimator in Ontario, Canada (2025).
Generate a clear, accurate collision repair estimate using 1 photo.
Return:

🛠 AI Damage Estimate

Severity: <minor/moderate/severe>
Estimated Cost: $<min> – $<max>
Areas: <list>
Damage Types: <list>

Short explanation.
""".strip()


async def estimate_damage(data_url: str, shop: Shop):
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": AI_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Shop: {shop.name}. Analyze the following image."},
                        {"type": "input_image", "image": data_url},
                    ],
                },
            ],
        )
        return response.output_text
    except Exception as e:
        print("AI error:", e)
        return "⚠️ Error analyzing image. Please resend your photo."


# ============================================================
# MAIN SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, background: BackgroundTasks):
    shop = get_shop_for_request(request)
    form = await request.form()

    body = (form.get("Body") or "").strip()
    from_number = form.get("From")
    to_number = form.get("To")
    num_media = int(form.get("NumMedia") or "0")
    media_url = form.get("MediaUrl0")

    resp = MessagingResponse()

    # VIN HANDLER ----------------------------------------------
    if body.upper().startswith("VIN:"):
        vin = body[4:].strip()
        decoded = await decode_vin(vin)

        if not decoded:
            resp.message("⚠️ Invalid VIN (must be 17 characters). Try: VIN: <VIN>")
            return Response(str(resp), media_type="application/xml")

        year = decoded.get("Model Year")
        make = decoded.get("Make")
        model = decoded.get("Model")
        trim = decoded.get("Trim")
        engine = decoded.get("Engine Model")

        msg = "🔍 VIN Data Found:\n\n"
        if year: msg += f"• Year: {year}\n"
        if make: msg += f"• Make: {make}\n"
        if model: msg += f"• Model: {model}\n"
        if trim: msg += f"• Trim: {trim}\n"
        if engine: msg += f"• Engine: {engine}\n"

        msg += "\nThis will help improve the accuracy of your estimate."
        resp.message(msg)
        return Response(str(resp), media_type="application/xml")

    # BOOKING HANDLER -------------------------------------------
    if body.upper().startswith("BOOK:"):
        text = body[5:].strip()
        add_booking_event(shop, text, from_number)
        resp.message("✅ Booking request sent! The shop will confirm shortly.")
        return Response(str(resp), media_type="application/xml")

    # PHOTO HANDLER ---------------------------------------------
    if num_media > 0 and media_url:
        resp.message(
            "📸 Thanks! We received your photos.\n"
            "Analyzing damage now…"
        )

        background.add_task(
            process_estimate_background,
            shop,
            media_url,
            from_number,
            to_number
        )

        return Response(str(resp), media_type="application/xml")

    # DEFAULT WELCOME -------------------------------------------
    resp.message(
        f"Welcome to {shop.name}!\n\n"
        "Send 1–3 photos of the damage for an instant AI estimate.\n\n"
        "To book an appointment after your estimate:\n"
        "BOOK: Name | Email | Vehicle | Preferred Time\n\n"
        "To decode your VIN:\n"
        "VIN: <17-digit VIN>"
    )
    return Response(str(resp), media_type="application/xml")


# ============================================================
# BACKGROUND ESTIMATE TASK
# ============================================================

async def process_estimate_background(shop, media_url, customer_number, twilio_number):
    try:
        created = datetime.utcnow()
        data_url = await download_media_as_data_url(media_url)

        if not data_url:
            twilio_client.messages.create(
                body="⚠️ Error processing your photos. Please resend.",
                from_=twilio_number,
                to=customer_number,
            )
            return

        estimate = await estimate_damage(data_url, shop)

        twilio_client.messages.create(
            body=(
                f"🛠 AI Damage Estimate for {shop.name}\n\n"
                f"{estimate}\n\n"
                "To book an appointment, reply:\n"
                "BOOK: Name | Email | Vehicle | Preferred Time"
            ),
            from_=twilio_number,
            to=customer_number,
        )

        add_lead_event(shop, estimate, customer_number, created)

    except Exception as exc:
        print("Background error:", exc)
        twilio_client.messages.create(
            body="⚠️ Error analyzing your images. Please resend a clear photo.",
            from_=twilio_number,
            to=customer_number
        )
