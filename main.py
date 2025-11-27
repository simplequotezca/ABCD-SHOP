import os
import json
import uuid
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from dateutil import parser as date_parser

from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
from googleapiclient.discovery import build
from google.oauth2 import service_account


# ============================================================
# ENV + CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "changeme-admin-key")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing.")

if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON missing.")


# ---- FIX: Prevent proxy injection crash (no 'proxies' arg) ----
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=None  # avoids Client.__init__(proxies=...) issue on Railway
)

app = FastAPI(title="Collision AI Estimator v2")


# ============================================================
# Load Multi-Shop Config
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str


def load_shops() -> Dict[str, Shop]:
    try:
        raw = json.loads(SHOPS_JSON)
        return {shop["webhook_token"]: Shop(**shop) for shop in raw}
    except Exception as e:
        print("ERROR loading SHOPS_JSON:", e)
        return {}


SHOPS = load_shops()


# ============================================================
# Google Calendar Auth
# ============================================================

def get_calendar_service():
    if not GOOGLE_SERVICE_JSON:
        raise RuntimeError("GOOGLE_SERVICE_JSON missing")

    info = json.loads(GOOGLE_SERVICE_JSON)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# ============================================================
# Simple In-Memory Conversation + Logging (Demo)
# ============================================================

# keyed by user phone number: { "stage": ..., "shop_token": ..., ... }
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

# simple log of estimates for admin view
ESTIMATE_LOG: list[Dict[str, Any]] = []


def reset_conversation(phone: str):
    CONVERSATIONS.pop(phone, None)


# ============================================================
# AI Estimator Function (v2 – more structured)
# ============================================================

async def ai_damage_estimate(image_b64: Optional[str], text: Optional[str]) -> Dict[str, Any]:
    """
    Returns a structured dict:
    {
      "severity": "Minor / Moderate / Severe",
      "estimated_cost_range": "e.g. $900 – $1,600",
      "parts": ["front bumper", "left fender"],
      "labor_hours": "8–12 hours",
      "repair_time": "2–3 business days",
      "detailed_explanation": "...",
      "risks": "...",
      "notes_for_shop": "..."
    }
    """

    system_prompt = """
You are an expert auto body estimator in Ontario, Canada (2025 pricing).
Estimate collision damage from images + text.

RETURN STRICT JSON ONLY with the exact keys:
severity (string: Minor/Moderate/Severe),
estimated_cost_range (string, CAD, e.g. "$1,200 – $2,000"),
parts (array of strings),
labor_hours (string, e.g. "6–9 hours"),
repair_time (string, e.g. "2–4 business days"),
detailed_explanation (string, 2–5 sentences),
risks (string, 1–3 sentences about hidden damage or safety),
notes_for_shop (string, advice for the body shop).

Base costs and labor on typical Ontario 2025 rates.
"""

    messages: list[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    user_content: list[Any] = []
    if text:
        user_content.append({"type": "input_text", "text": text.strip()})
    else:
        user_content.append({"type": "input_text", "text": "Customer sent damage photos for estimate."})

    if image_b64:
        user_content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image_b64}"
        })

    messages.append({"role": "user", "content": user_content})

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.25,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        # basic sanity defaults
        data.setdefault("severity", "Moderate")
        data.setdefault("estimated_cost_range", "Request shop inspection")
        data.setdefault("parts", [])
        data.setdefault("labor_hours", "N/A")
        data.setdefault("repair_time", "N/A")
        data.setdefault("detailed_explanation", "")
        data.setdefault("risks", "")
        data.setdefault("notes_for_shop", "")
        return data

    except Exception as e:
        print("AI Error:", e)
        return {
            "severity": "Unknown",
            "estimated_cost_range": "N/A",
            "parts": [],
            "labor_hours": "N/A",
            "repair_time": "N/A",
            "detailed_explanation": "Error generating estimate.",
            "risks": "",
            "notes_for_shop": ""
        }


def format_estimate_for_sms(data: Dict[str, Any], shop_name: str) -> str:
    parts_str = ", ".join(data.get("parts") or []) or "Not clearly visible from photos."

    sms = (
        f"AI Damage Estimate for {shop_name}\n\n"
        f"Severity: {data.get('severity')}\n"
        f"Estimated Cost (Ontario 2025): {data.get('estimated_cost_range')}\n"
        f"Labor: {data.get('labor_hours')}\n"
        f"Repair Time: {data.get('repair_time')}\n"
        f"Affected Parts: {parts_str}\n\n"
        f"{data.get('detailed_explanation')}\n\n"
        f"Risks: {data.get('risks')}\n\n"
        f"Note for shop: {data.get('notes_for_shop')}\n\n"
        f"To book a visit and lock this into the calendar, reply:\n"
        f"BOOK"
    )
    return sms


# ============================================================
# Create Google Calendar Appointment
# ============================================================

def parse_customer_datetime(text: str) -> datetime:
    """
    Try to parse customer natural language date/time.
    Fallback: next business day at 10 AM UTC.
    """
    try:
        dt = date_parser.parse(text, fuzzy=True)
        if dt.tzinfo:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        return dt
    except Exception:
        # fallback: tomorrow 10:00
        base = datetime.utcnow() + timedelta(days=1)
        return base.replace(hour=10, minute=0, second=0, microsecond=0)


def create_calendar_event(shop: Shop, user_data: dict) -> str:
    service = get_calendar_service()

    start_dt = parse_customer_datetime(user_data.get("preferred_datetime", ""))
    end_dt = start_dt + timedelta(minutes=30)

    start_iso = start_dt.isoformat() + "Z"
    end_iso = end_dt.isoformat() + "Z"

    description = (
        f"Name: {user_data.get('name')}\n"
        f"Phone: {user_data.get('phone')}\n"
        f"Email: {user_data.get('email')}\n"
        f"Vehicle: {user_data.get('vehicle')}\n\n"
        f"Estimate:\n{user_data.get('estimate_text')}\n\n"
        f"AI Raw Data:\n{json.dumps(user_data.get('estimate_struct'), indent=2)}\n\n"
        f"Image URL (Twilio): {user_data.get('image_url')}"
    )

    event_body = {
        "summary": f"AI Damage Estimate - {user_data.get('name')}",
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }

    event = service.events().insert(
        calendarId=shop.calendar_id,
        body=event_body
    ).execute()

    return event.get("htmlLink", "")


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "service": "Collision AI Estimator v2"}


@app.get("/admin/estimates")
def admin_estimates(token: str):
    if token != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return JSONResponse(ESTIMATE_LOG)


# ============================================================
# Twilio Webhook – full flow
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):

    form = await request.form()
    token = request.query_params.get("token")

    if token not in SHOPS:
        return PlainTextResponse("Invalid shop token", status_code=400)

    shop = SHOPS[token]

    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or "unknown"
    image_url = form.get("MediaUrl0")  # Twilio stores public URL – keep it for calendar

    # Ensure conversation container exists
    convo = CONVERSATIONS.get(from_number)

    # 1) If user is mid-booking flow
    if convo and convo.get("stage") in {"get_name", "get_email", "get_vehicle", "get_datetime"}:
        stage = convo["stage"]
        reply = MessagingResponse()

        if stage == "get_name":
            convo["name"] = body
            convo["stage"] = "get_email"
            reply.message("Thanks! Please reply with your EMAIL address.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        if stage == "get_email":
            convo["email"] = body
            convo["stage"] = "get_vehicle"
            reply.message("Got it. What vehicle are you driving? (Year, Make, Model, colour)")
            return PlainTextResponse(str(reply), media_type="application/xml")

        if stage == "get_vehicle":
            convo["vehicle"] = body
            convo["stage"] = "get_datetime"
            reply.message("Last step – what day & time works best for you?\n\nExample: \"Tuesday at 3pm\"")
            return PlainTextResponse(str(reply), media_type="application/xml")

        if stage == "get_datetime":
            convo["preferred_datetime"] = body

            # Build calendar payload
            user_data = {
                "name": convo.get("name"),
                "phone": from_number,
                "email": convo.get("email"),
                "vehicle": convo.get("vehicle"),
                "preferred_datetime": convo.get("preferred_datetime"),
                "estimate_text": convo.get("estimate_text"),
                "estimate_struct": convo.get("estimate_struct"),
                "image_url": convo.get("image_url"),
            }

            try:
                event_link = create_calendar_event(shop, user_data)
                reply.message(
                    f"You're booked with {shop.name}! 🎉\n\n"
                    f"We've added your visit to the calendar.\n"
                    f"Event link (for the shop): {event_link}\n\n"
                    f"If you need to change anything, reply with your updates anytime."
                )
            except Exception as e:
                print("Calendar error:", e)
                reply.message(
                    "We had an issue booking the calendar, but your info was received. "
                    "The shop will contact you to confirm a time."
                )

            # end conversation
            reset_conversation(from_number)
            return PlainTextResponse(str(reply), media_type="application/xml")

    # 2) New message or not in booking flow – process estimate
    reply = MessagingResponse()

    # Download + encode image (optional)
    image_b64 = None
    if image_url:
        try:
            img_bytes = requests.get(image_url).content
            image_b64 = base64.b64encode(img_bytes).decode()
        except Exception as e:
            print("Image download error:", e)
            image_b64 = None

    # Run AI estimator
    estimate_struct = await ai_damage_estimate(image_b64, body)
    estimate_sms = format_estimate_for_sms(estimate_struct, shop.name)

    # Log for admin
    ESTIMATE_LOG.append({
        "id": str(uuid.uuid4()),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "from": from_number,
        "shop_id": shop.id,
        "shop_name": shop.name,
        "text": body,
        "image_url": image_url,
        "estimate": estimate_struct,
    })

    # Store conversation context for optional booking
    CONVERSATIONS[from_number] = {
        "stage": "offer_book",  # waiting to see if they text BOOK
        "shop_token": token,
        "image_url": image_url,
        "estimate_struct": estimate_struct,
        "estimate_text": estimate_sms,
    }

    # If user just wrote "BOOK" without going through estimate, handle below,
    # but first send the estimate result for normal flow:
    if body.strip().upper() == "BOOK":
        # fall through to booking start below
        pass

    # send estimate to customer
    reply.message(estimate_sms)

    return PlainTextResponse(str(reply), media_type="application/xml")


@app.post("/sms-webhook/book")
async def sms_force_book(request: Request):
    """
    Optional 2nd webhook if you ever want a dedicated "BOOK" number.
    Not required for demo – /sms-webhook main route already handles everything.
    """
    return PlainTextResponse("OK")
