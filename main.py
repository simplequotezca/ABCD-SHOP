import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto-estimator")

app = FastAPI()

# ============================================================
# Environment
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

SHOPS_JSON = os.getenv("SHOPS_JSON")
if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON env var is required")

try:
    SHOPS_CONFIG: List[Dict[str, Any]] = json.loads(SHOPS_JSON)
except Exception as e:
    raise RuntimeError(f"SHOPS_JSON is not valid JSON: {e}")

# IMPORTANT: must match your Railway env var name
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Dict[str, Any] = {}
    hours: Dict[str, Any] = {}


def load_shops() -> Dict[str, Shop]:
    shops: Dict[str, Shop] = {}
    for raw in SHOPS_CONFIG:
        shop = Shop(**raw)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()

# ============================================================
# Simple in-memory state per phone number
# ============================================================

class Session(BaseModel):
    stage: str = "start"          # "start" | "estimate_sent" | "booking_pending"
    last_estimate: Optional[Dict[str, Any]] = None
    shop_token: Optional[str] = None


SESSIONS: Dict[str, Session] = {}


def get_session(phone: str) -> Session:
    s = SESSIONS.get(phone)
    if not s:
        s = Session()
        SESSIONS[phone] = s
    return s


# ============================================================
# Google Calendar helpers
# ============================================================

def get_calendar_service():
    """
    Builds a Google Calendar API client using the GOOGLE_SERVICE_ACCOUNT_JSON env var.
    """
    if not GOOGLE_SERVICE_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var is missing")

    try:
        info = json.loads(GOOGLE_SERVICE_JSON)
    except Exception as e:
        raise RuntimeError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    service = build("calendar", "v3", credentials=creds)
    return service


def create_simple_booking(shop: Shop, customer_name: str, customer_phone: str) -> Optional[str]:
    """
    Very simple booking: create a 1-hour event tomorrow at ~10am Toronto
    in the shop's calendar.
    Returns Google event htmlLink or None on failure.
    """
    if not shop.calendar_id:
        logger.warning("Shop %s has no calendar_id", shop.id)
        return None

    try:
        service = get_calendar_service()
    except Exception as e:
        logger.exception("Failed to init Google Calendar: %s", e)
        return None

    # crude but fine: next day, 15:00 UTC ≈ 10:00 Toronto depending on DST
    start_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    start_dt = start_dt.replace(hour=15)
    end_dt = start_dt + timedelta(hours=1)

    event_body = {
        "summary": f"AI Estimate Booking – {customer_name}",
        "description": (
            f"Auto-body AI estimate lead.\n"
            f"Customer: {customer_name}\n"
            f"Phone: {customer_phone}"
        ),
        "start": {"dateTime": start_dt.isoformat() + "Z"},
        "end": {"dateTime": end_dt.isoformat() + "Z"},
    }

    try:
        event = service.events().insert(calendarId=shop.calendar_id, body=event_body).execute()
        return event.get("htmlLink")
    except Exception as e:
        logger.exception("Failed to create calendar event: %s", e)
        return None


# ============================================================
# AI damage estimator
# ============================================================

def call_ai_estimator(
    shop: Shop,
    text: str,
    image_url: Optional[str],
) -> Dict[str, Any]:
    """
    Calls GPT-4.1-mini vision to produce a structured estimate.
    On any error, returns a safe fallback dictionary.
    """
    system_prompt = f"""
You are an auto-body damage estimator for {shop.name} in Ontario, Canada (2025 prices).

You will receive:
- Optional photo of vehicle damage
- Text description from the customer
- Shop's pricing config and hours

You must respond ONLY with JSON in this exact structure:

{{
  "severity": "minor|moderate|severe",
  "price_range": [min_total, max_total],
  "confidence": 0-100,
  "labor_hours": "string like '4-6 hours'",
  "turnaround_time": "string like '2-4 business days'",
  "parts": ["short list of likely parts affected"],
  "explanation": "2-4 sentences in plain language for the customer",
  "notes_for_shop": "1-3 sentences of technical notes for the body shop"
}}

Use the shop's pricing config and typical Ontario labor/material rates.
If information is missing or the photo is unclear, be conservative and lower confidence.
"""

    pricing_str = json.dumps(shop.pricing, indent=2)
    hours_str = json.dumps(shop.hours, indent=2)

    user_text = (
        "Customer description:\n"
        f"{text or '(no description)'}\n\n"
        f"Shop pricing JSON:\n{pricing_str}\n\n"
        f"Shop business hours JSON:\n{hours_str}\n"
        "Estimate total cost in CAD."
    )

    content: List[Any] = [{"type": "text", "text": user_text}]
    if image_url:
        content.append(
            {
                "type": "input_image",
                "image_url": {"url": image_url},
            }
        )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, list):
            combined = "".join(
                part.get("text", "") for part in raw if isinstance(part, dict)
            )
        else:
            combined = str(raw)

        combined = combined.strip()
        start = combined.find("{")
        end = combined.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = combined[start : end + 1]
        else:
            json_str = combined

        data = json.loads(json_str)
        if "price_range" not in data or not isinstance(data["price_range"], list):
            raise ValueError("price_range missing")
        return data
    except Exception as e:
        logger.exception("AI estimator failed, using fallback: %s", e)
        return {
            "severity": "moderate",
            "price_range": [900, 1600],
            "confidence": 40,
            "labor_hours": "4–6 hours (rough guess)",
            "turnaround_time": "2–4 business days",
            "parts": ["front / rear panels", "bumper", "paint & materials"],
            "explanation": (
                "Based on a typical collision in Ontario, a moderate repair often falls in this range. "
                "The final price may be higher or lower after an in-person inspection."
            ),
            "notes_for_shop": (
                "Fallback estimate used because the AI model could not be reached or parsing failed."
            ),
        }


# ============================================================
# Twilio SMS webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request) -> PlainTextResponse:
    """
    Main Twilio webhook.

    Supports optional `token` query param for multi-shop routing, but falls back
    to the first shop in SHOPS_JSON if missing.

    ALWAYS returns valid TwiML, even on internal errors (so Twilio never sees 5xx).
    """
    resp = MessagingResponse()

    try:
        form = await request.form()
        from_number = form.get("From", "")
        body = (form.get("Body") or "").strip()
        media_url = form.get("MediaUrl0") or None

        # Which shop?
        token = request.query_params.get("token")
        shop: Optional[Shop] = None
        if token and token in SHOPS_BY_TOKEN:
            shop = SHOPS_BY_TOKEN[token]
        else:
            # default: first shop
            any_shop_token = next(iter(SHOPS_BY_TOKEN.keys()))
            shop = SHOPS_BY_TOKEN[any_shop_token]

        session = get_session(from_number)
        session.shop_token = shop.webhook_token

        logger.info(
            "Incoming SMS from %s | body=%r | media=%r | stage=%s",
            from_number,
            body,
            media_url,
            session.stage,
        )

        # ---------------- Stage: start ----------------
        if session.stage == "start":
            if not body and not media_url:
                msg = (
                    f"Thanks for contacting {shop.name}.\n\n"
                    "Please reply with a brief description of the damage and "
                    "optionally attach clear photos of the affected areas."
                )
                resp.message(msg)
                return PlainTextResponse(str(resp), media_type="application/xml")

            estimate = call_ai_estimator(shop, text=body, image_url=media_url)
            session.last_estimate = estimate
            session.stage = "estimate_sent"

            sev = estimate.get("severity", "unknown").title()
            pr_min, pr_max = estimate.get("price_range", [0, 0])
            expl = estimate.get("explanation", "")
            conf = estimate.get("confidence", 0)

            msg_lines = [
                f"AI Damage Estimate for {shop.name}",
                "",
                f"Severity: {sev}",
                f"Estimated Cost (Ontario 2025): ${pr_min:,.0f} – ${pr_max:,.0f} + HST",
                f"Confidence: {conf}%",
            ]
            if expl:
                msg_lines.append("")
                msg_lines.append(expl)

            msg_lines.append("")
            msg_lines.append("Reply 1 to book an appointment, or 2 to ask another question.")

            resp.message("\n".join(msg_lines))
            return PlainTextResponse(str(resp), media_type="application/xml")

        # ---------------- Stage: estimate_sent ----------------
        if session.stage == "estimate_sent":
            normalized = body.lower()
            if normalized in {"1", "book", "book now", "book appointment"}:
                session.stage = "booking_pending"
                resp.message(
                    "Great! To book your free estimate, please reply with your full name.\n\n"
                    "Example: John Smith"
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            if normalized in {"2", "restart", "new"}:
                session.stage = "start"
                resp.message(
                    "No problem. Please send a new description of the damage and photos if possible."
                )
                return PlainTextResponse(str(resp), media_type="application/xml")

            # treat as follow-up question
            resp.message(
                "If you’d like to book an appointment reply 1.\n"
                "To restart with new photos / details reply 2."
            )
            return PlainTextResponse(str(resp), media_type="application/xml")

        # ---------------- Stage: booking_pending ----------------
        if session.stage == "booking_pending":
            customer_name = body.strip()
            if not customer_name:
                resp.message("Please reply with your full name so we can book your appointment.")
                return PlainTextResponse(str(resp), media_type="application/xml")

            shop = SHOPS_BY_TOKEN.get(session.shop_token or "", None) or shop
            event_link = create_simple_booking(shop, customer_name, from_number)

            if event_link:
                msg = (
                    f"Thanks {customer_name}! We’ve booked a 1-hour estimate slot for you.\n\n"
                    f"Calendar link: {event_link}\n\n"
                    f"{shop.name} will confirm by text or phone if any changes are needed."
                )
            else:
                msg = (
                    f"Thanks {customer_name}! {shop.name} has received your details and will call "
                    "or text you to confirm the best time for your estimate."
                )

            # reset so future messages start fresh
            SESSIONS.pop(from_number, None)

            resp.message(msg)
            return PlainTextResponse(str(resp), media_type="application/xml")

        # ---------------- Fallback: reset ----------------
        SESSIONS.pop(from_number, None)
        resp.message(
            "Let’s start over.\n\nPlease send a short description of the damage and "
            "photos if possible."
        )
        return PlainTextResponse(str(resp), media_type="application/xml")

    except Exception as e:
        logger.exception("Unhandled error in sms_webhook: %s", e)
        # Never let Twilio see a 5xx
        resp.message(
            "Sorry – our AI estimator had an issue processing your message. "
            "Please try again, or call the shop directly."
        )
        return PlainTextResponse(str(resp), media_type="application/xml")
