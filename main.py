# ============================================================
# FULL main.py — WORKING AI ESTIMATOR + GOOGLE CALENDAR
# ============================================================

import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# ============================================================
# ENVIRONMENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin123")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ============================================================
# LOAD SHOPS
# ============================================================

class Shop:
    def __init__(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.name = data["name"]
        self.webhook_token = data["webhook_token"]
        self.calendar_id = data.get("calendar_id")  # Google Calendar ID

def load_shops() -> Dict[str, Shop]:
    try:
        data = json.loads(SHOPS_JSON)
        return {s["webhook_token"]: Shop(s) for s in data}
    except Exception as e:
        print("Failed to load SHOPS_JSON:", e)
        return {}

SHOPS = load_shops()

# Memory store for AI estimates (for booking later)
LAST_ESTIMATES: Dict[str, Dict[str, Any]] = {}

# ============================================================
# GOOGLE CALENDAR SERVICE (INSERT EVENT)
# ============================================================

from google.oauth2 import service_account
from googleapiclient.discovery import build

GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")

def get_google_service():
    if not GOOGLE_CREDS_BASE64:
        print("No Google credentials found.")
        return None

    import base64
    creds_json = base64.b64decode(GOOGLE_CREDS_BASE64)

    creds = service_account.Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)

calendar_service = get_google_service()

def create_calendar_event(shop: Shop, summary: str, description: str):
    if not shop.calendar_id or not calendar_service:
        print("[WARN] Calendar not configured for", shop.name)
        return

    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": datetime.utcnow().isoformat() + "Z"},
        "end": {"dateTime": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"},
    }

    try:
        calendar_service.events().insert(
            calendarId=shop.calendar_id,
            body=event
        ).execute()
        print("[OK] Calendar event created.")
    except Exception as e:
        print("[ERROR] Calendar insert failed:", e)

# ============================================================
# HELPERS
# ============================================================

async def download_image_as_data_url(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            import base64
            b64 = base64.b64encode(r.content).decode()
            mime = r.headers.get("content-type", "image/jpeg")
            return f"data:{mime};base64,{b64}"
    except Exception as e:
        print("Download error:", e)
        return None

async def generate_damage_estimate(image_data_urls: List[str], shop_name: str) -> str:
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze auto collision damage for {shop_name}."}
            ] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data_urls]
        }]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )

        return completion.choices[0].message["content"]

    except Exception as e:
        print("AI ERROR:", e)
        return "⚠️ AI Processing Error: Please try again in a few minutes."

def parse_booking_message(body: str) -> Dict[str, str]:
    lines = body.split("\n")
    data = {}
    for l in lines:
        if ":" in l:
            k, v = l.split(":", 1)
            data[k.strip().lower()] = v.strip()
    return data

def get_shop_from_request(request: Request) -> Shop:
    token = request.query_params.get("token")
    return SHOPS.get(token)

# ============================================================
# MAIN WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    shop = get_shop_from_request(request)
    if not shop:
        resp = MessagingResponse()
        resp.message("Shop not found. Invalid token.")
        return Response(str(resp), media_type="application/xml")

    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or "Unknown"
    num_media = int(form.get("NumMedia") or "0")

    resp = MessagingResponse()

    # =======================================================
    # BOOKING
    # =======================================================
    if body.upper().startswith("BOOKING:"):
        customer = parse_booking_message(body)
        stored = LAST_ESTIMATES.get(from_number, {})

        description = (
            f"Name: {customer.get('name')}\n"
            f"Phone: {customer.get('phone')}\n"
            f"Email: {customer.get('email')}\n"
            f"Preferred: {customer.get('preferred date & time')}\n\n"
            f"AI Estimate:\n{stored.get('estimate_text')}"
        )

        create_calendar_event(
            shop,
            summary=f"Repair Booking — {customer.get('name')}",
            description=description
        )

        resp.message(
            "✅ Your appointment request has been submitted.\n"
            f"{shop.name} will confirm shortly."
        )
        return Response(str(resp), media_type="application/xml")

    # =======================================================
    # OPTION 1 → BOOK APPOINTMENT
    # =======================================================
    if body == "1":
        msg = (
            "To book your repair appointment, reply in this format:\n\n"
            "BOOKING:\n"
            "Name: John Doe\n"
            "Phone: 647-000-0000\n"
            "Email: example@email.com\n"
            "Preferred Date & Time: Tues Dec 5 @ 3pm\n"
            "Vehicle: 2018 Honda Civic (blue)"
        )
        resp.message(msg)
        return Response(str(resp), media_type="application/xml")

    # =======================================================
    # OPTION 2 → MORE PHOTOS
    # =======================================================
    if body == "2":
        resp.message(
            "No problem! Please send 1–3 more photos from different angles."
        )
        return Response(str(resp), media_type="application/xml")

    # =======================================================
    # IMAGES RECEIVED → RUN AI
    # =======================================================
    if num_media > 0:
        resp.message(
            "📸 Thanks! We received your photos.\n"
            "Analyzing damage now…"
        )

        img_data_urls = []
        img_raw_urls = []

        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            if not url:
                continue
            img_raw_urls.append(url)
            data_url = await download_image_as_data_url(url)
            if data_url:
                img_data_urls.append(data_url)

        if not img_data_urls:
            resp.message("⚠️ Couldn't process the images. Please resend.")
            return Response(str(resp), media_type="application/xml")

        estimate = await generate_damage_estimate(img_data_urls, shop.name)

        resp.message(estimate)

        LAST_ESTIMATES[from_number] = {
            "shop_id": shop.id,
            "estimate_text": estimate,
            "image_urls": img_raw_urls,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log lead into Google Calendar (even without appointment)
        create_calendar_event(
            shop,
            summary=f"AI Estimate Lead ({from_number})",
            description=estimate
        )

        return Response(str(resp), media_type="application/xml")

    # =======================================================
    # DEFAULT WELCOME MESSAGE
    # =======================================================
    welcome = (
        f"Welcome to {shop.name}!\n\n"
        "Send 1–3 clear photos of the damage to receive a fast AI estimate "
        "and an option to book your repair appointment."
    )

    resp.message(welcome)
    return Response(str(resp), media_type="application/xml")
