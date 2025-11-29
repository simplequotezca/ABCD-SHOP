import os
import json
import re
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# Google Calendar imports (Option B: full integration enabled)
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()


# ============================================================
# ENV + GLOBALS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "changeme_admin")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Shops configuration from SHOPS_JSON
SHOPS_JSON = os.getenv("SHOPS_JSON")
if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON env var is required")

try:
    SHOPS: List[Dict[str, Any]] = json.loads(SHOPS_JSON)
except Exception as e:
    raise RuntimeError(f"Failed to parse SHOPS_JSON: {e}")

# Build lookup by webhook token
SHOPS_BY_TOKEN: Dict[str, Dict[str, Any]] = {}
for shop in SHOPS:
    token = shop.get("webhook_token")
    if token:
        SHOPS_BY_TOKEN[token] = shop


# Google Calendar service (Option B: enabled)
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_google_credentials():
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON not set")
    try:
        info = json.loads(GOOGLE_CREDS_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=GOOGLE_SCOPES
        )
        return creds
    except Exception as e:
        raise RuntimeError(f"Failed to create Google credentials: {e}")


def get_calendar_service():
    creds = get_google_credentials()
    return build("calendar", "v3", credentials=creds)


# ============================================================
# Helper: basic text cleaners / parsers
# ============================================================

def clean_phone_number(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"+1{digits}"
    if not digits.startswith("+"):
        return "+" + digits
    return digits


def extract_estimate_confirmation(message: str) -> Optional[bool]:
    """
    Very simple yes/no detector for booking confirmation.
    Returns True if user clearly wants to book, False if clearly no, else None.
    """
    text = (message or "").lower()
    yes_words = ["yes", "yeah", "yup", "ok", "okay", "book", "schedule", "confirm"]
    no_words = ["no", "nope", "cancel", "later", "not now", "don't"]

    if any(w in text for w in yes_words):
        return True
    if any(w in text for w in no_words):
        return False
    return None


def extract_preferred_time(message: str) -> Optional[str]:
    """
    Heuristic: try to pull a rough time phrase like 'tomorrow at 3', 'monday 10am', etc.
    This is deliberately simple – AI will refine.
    """
    # For now just return the full message as a hint.
    return message.strip() if message else None


# ============================================================
# AI ESTIMATOR LOGIC
# ============================================================

def build_image_payload_from_twilio(form: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build OpenAI 'image_url' content parts from Twilio MediaUrl fields.
    """
    media_parts: List[Dict[str, Any]] = []
    num_media = int(form.get("NumMedia", "0") or 0)

    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        if not media_url:
            continue
        # Twilio gives us a public URL – pass directly to OpenAI as remote URL
        media_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": media_url,
                },
            }
        )
    return media_parts


def build_ai_messages(
    shop: Dict[str, Any],
    user_text: str,
    image_parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build messages for the OpenAI Chat Completions API.
    """

    pricing = shop.get("pricing", {})
    labor_rates = pricing.get("labor_rates", {})
    materials_rate = pricing.get("materials_rate")
    base_floor = pricing.get("base_floor", {})

    system_prompt = f"""
You are an AI auto body damage estimator assistant for {shop.get("name")}.
Location: Ontario, Canada (2025 pricing).
Your job:

1. Analyse any provided vehicle damage photos and/or text description.
2. Decide damage severity: Minor, Moderate, or Severe.
3. Provide a realistic repair cost range based on this shop's pricing:
   - Labor rates: body = {labor_rates.get("body")}, paint = {labor_rates.get("paint")}
   - Materials rate: {materials_rate} per hour
   - Base floor ranges (CAD):
       Minor: {base_floor.get("minor_min")} – {base_floor.get("minor_max")}
       Moderate: {base_floor.get("moderate_min")} – {base_floor.get("moderate_max")}
       Severe: {base_floor.get("severe_min")} – {base_floor.get("severe_max")}

4. Identify:
   - Affected panels/areas (e.g., front bumper, hood, left fender, etc.).
   - Damage types (e.g., scratch, dent, deep dent, crack, deformation, misalignment).

5. Output a **short, SMS-friendly** response with:
   - One-line summary
   - Severity
   - Estimated cost range (CAD)
   - Key damaged areas
   - 1–2 sentence explanation
   - Clear note: "This is a preliminary estimate based on photos and may change after in-person inspection."

Always keep it under ~5 SMS-length messages (about 700 characters total).
Don't oversell; be honest and realistic, but not pessimistic.
    """.strip()

    user_content: List[Dict[str, Any]] = []

    if image_parts:
        user_content.extend(image_parts)

    text_for_model = user_text or "Customer did not provide a text description."
    user_content.append(
        {
            "type": "text",
            "text": f"Customer text description: {text_for_model}",
        }
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


def run_ai_estimate(
    shop: Dict[str, Any],
    user_text: str,
    image_parts: List[Dict[str, Any]],
) -> str:
    """
    Call OpenAI with vision + text to generate an estimate summary.
    """
    messages = build_ai_messages(shop, user_text, image_parts)

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=400,
        temperature=0.4,
    )

    ai_text = completion.choices[0].message.content
    return ai_text.strip()


# ============================================================
# GOOGLE CALENDAR BOOKING
# ============================================================

def book_calendar_event(
    shop: Dict[str, Any],
    customer_name: str,
    customer_phone: str,
    customer_email: Optional[str],
    vehicle_info: Optional[str],
    estimate_summary: str,
    preferred_time_note: Optional[str],
) -> Optional[str]:
    """
    Create a Google Calendar event in the shop's calendar.
    Returns the event HTML link if successful, else None.
    """

    calendar_id = shop.get("calendar_id")
    if not calendar_id:
        return None

    service = get_calendar_service()

    # Simple heuristic: book 24 hours from now for 45 minutes if no clear time
    start = datetime.utcnow() + timedelta(days=1)
    end = start + timedelta(minutes=45)

    description_lines = [
        f"AI Damage Estimate Lead for {shop.get('name')}",
        "",
        f"Customer: {customer_name}",
        f"Phone: {customer_phone}",
    ]
    if customer_email:
        description_lines.append(f"Email: {customer_email}")
    if vehicle_info:
        description_lines.append(f"Vehicle: {vehicle_info}")
    if preferred_time_note:
        description_lines.append(f"Customer mentioned time: {preferred_time_note}")
    description_lines.append("")
    description_lines.append("AI estimate summary:")
    description_lines.append(estimate_summary)

    event = {
        "summary": f"AI Estimate – {customer_name}",
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": start.isoformat() + "Z",
        },
        "end": {
            "dateTime": end.isoformat() + "Z",
        },
    }

    created = service.events().insert(calendarId=calendar_id, body=event).execute()
    return created.get("htmlLink")


# ============================================================
# TWILIO WEBHOOK HANDLER
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio will POST form-encoded data here.
    We use the `token` query param to identify which shop this is for.
    """
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token)

    # Always respond TwiML, even on failures
    reply = MessagingResponse()

    if not shop:
        reply.message(
            "Sorry, this number is not configured correctly yet. "
            "Please contact the shop directly."
        )
        return Response(content=str(reply), media_type="application/xml")

    form = await request.form()
    user_message = form.get("Body", "").strip()
    from_number = form.get("From")

    # Twilio's 'From' is the customer's number
    cleaned_phone = clean_phone_number(from_number)

    try:
        # Build image parts for AI if any
        image_parts = build_image_payload_from_twilio(form)

        # Run the AI estimator
        estimate_text = run_ai_estimate(shop, user_message, image_parts)

        # Try to detect if the user is confirming a booking
        confirmation = extract_estimate_confirmation(user_message)
        preferred_time_note = extract_preferred_time(user_message)

        if confirmation is True:
            # Very simple flow – in a real system you'd persist contact first
            customer_name = "Customer"
            customer_email = None
            vehicle_info = None

            try:
                event_link = book_calendar_event(
                    shop=shop,
                    customer_name=customer_name,
                    customer_phone=cleaned_phone,
                    customer_email=customer_email,
                    vehicle_info=vehicle_info,
                    estimate_summary=estimate_text,
                    preferred_time_note=preferred_time_note,
                )
                if event_link:
                    reply.message(
                        f"You're booked! We've scheduled an appointment at "
                        f"{shop.get('name')}. We'll confirm exact time by text shortly.\n\n"
                        f"Estimate:\n{estimate_text}"
                    )
                else:
                    reply.message(
                        f"Estimate:\n{estimate_text}\n\n"
                        "We tried to book an appointment but couldn't access the shop's calendar. "
                        "Someone from the shop will contact you to finalize a time."
                    )
            except Exception:
                # Calendar error: send estimate only
                reply.message(
                    f"Estimate:\n{estimate_text}\n\n"
                    "We couldn't auto-book a time, but the shop will follow up to schedule you in."
                )
        else:
            # Just send the estimate + call to action
            reply.message(
                f"{estimate_text}\n\n"
                "If you'd like to book an in-person inspection, reply 'BOOK' with your preferred day/time."
            )

    except Exception:
        # Safety net: never let the webhook crash Twilio
        reply.message(
            "Sorry, something went wrong while generating your estimate. "
            "Please send a clear photo of the damage and a brief description, "
            "or call the shop directly."
        )

    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# BASIC HEALTH + ADMIN ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "shops_configured": list(SHOPS_BY_TOKEN.keys())}


class ShopInfo(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str]


@app.get("/admin/shops", response_model=List[ShopInfo])
def list_shops(admin_key: str):
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return [
        ShopInfo(
            id=s.get("id", ""),
            name=s.get("name", ""),
            webhook_token=s.get("webhook_token", ""),
            calendar_id=s.get("calendar_id"),
        )
        for s in SHOPS_BY_TOKEN.values()
    ]
