import os
import json
import uuid
import base64
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, EmailStr
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from openai import OpenAI

from google.oauth2 import service_account
from googleapiclient.discovery import build as build_gcal

# ============================================================
# FastAPI
# ============================================================

app = FastAPI()

# ============================================================
# Environment
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON is required (multi-shop config)")

# OpenAI client (simple, no proxies/http_client args)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Twilio client
twilio_client: Optional[TwilioClient] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ============================================================
# Google Calendar helper
# ============================================================

def get_gcal_service() -> Optional[Any]:
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        return None

    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build_gcal("calendar", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        print(f"[GCAL] Failed to build service: {e}")
        return None


gcal_service = get_gcal_service()

# ============================================================
# Shop config models
# ============================================================

class PricingFloor(BaseModel):
    minor_min: float
    minor_max: float
    moderate_min: float
    moderate_max: float
    severe_min: float
    severe_max: float


class LaborRates(BaseModel):
    body: float
    paint: float


class PricingConfig(BaseModel):
    labor_rates: LaborRates
    materials_rate: float
    base_floor: PricingFloor


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Optional[PricingConfig] = None
    hours: Dict[str, List[str]] = {}


def load_shops() -> Dict[str, Shop]:
    """Load shops from SHOPS_JSON env and index by webhook_token."""
    try:
        data = json.loads(SHOPS_JSON)
        shops = {}
        for raw in data:
            shop = Shop(**raw)
            shops[shop.webhook_token] = shop
        return shops
    except Exception as e:
        raise RuntimeError(f"Invalid SHOPS_JSON: {e}")


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()


def get_shop_from_request(request: Request) -> Shop:
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing ?token=shop_token")
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404, detail="Unknown shop token")
    return shop


# ============================================================
# Utility helpers
# ============================================================

def normalize_phone(phone: str) -> str:
    """Normalize phone to E.164-ish for Twilio lookups."""
    digits = re.sub(r"\D", "", phone or "")
    if digits.startswith("1") and len(digits) == 11:
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if digits.startswith("+"):
        return digits
    return "+" + digits


def parse_damage_keywords(text: str) -> Dict[str, Any]:
    """Very light parsing to feed into the LLM prompt."""
    text_lower = (text or "").lower()

    areas = []
    for area in ["front", "rear", "side", "left", "right", "hood", "trunk", "roof", "bumper", "door", "fender"]:
        if area in text_lower:
            areas.append(area)

    damage_types = []
    for d in ["scratch", "dent", "deep dent", "crack", "broken light", "paint peel", "rust"]:
        if d in text_lower:
            damage_types.append(d)

    return {"areas": areas, "damage_types": damage_types}


def estimate_cost_from_llm_json(shop: Shop, severity: str, llm_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine the model's structured JSON with shop pricing config
    to give a realistic Ontario-style estimate range.
    """
    pricing = shop.pricing
    if not pricing:
        # No custom pricing, just return model's suggestion
        return {
            "severity": severity,
            "estimated_cost_min": llm_json.get("estimated_cost_min"),
            "estimated_cost_max": llm_json.get("estimated_cost_max"),
        }

    floor = pricing.base_floor

    if severity == "minor":
        base_min, base_max = floor.minor_min, floor.minor_max
    elif severity == "moderate":
        base_min, base_max = floor.moderate_min, floor.moderate_max
    else:
        base_min, base_max = floor.severe_min, floor.severe_max

    llm_min = llm_json.get("estimated_cost_min") or base_min
    llm_max = llm_json.get("estimated_cost_max") or base_max

    est_min = max(base_min, float(llm_min))
    est_max = max(est_min + 100, float(llm_max))

    return {
        "severity": severity,
        "estimated_cost_min": round(est_min),
        "estimated_cost_max": round(est_max),
    }


def build_appointment_slots(shop: Shop, days_ahead: int = 7) -> List[Dict[str, Any]]:
    """
    Build simple “next X days” slots based on shop.hours (mon-sun).
    We don't read existing events — this is just candidate times
    to show the user and then actually book into Google Calendar.
    """
    slots: List[Dict[str, Any]] = []
    now = datetime.now()
    for offset in range(days_ahead):
        day = now + timedelta(days=offset)
        weekday = day.strftime("%A").lower()  # monday, tuesday, etc.

        ranges = shop.hours.get(weekday, [])
        for r in ranges:
            # "9am-5pm" -> start, end
            m = re.match(r"(\d{1,2})(am|pm)-(\d{1,2})(am|pm)", r.replace(" ", "").lower())
            if not m:
                continue
            sh, sa, eh, ea = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)

            if sa == "pm" and sh != 12:
                sh += 12
            if ea == "pm" and eh != 12:
                eh += 12

            start_dt = day.replace(hour=sh, minute=0, second=0, microsecond=0)
            end_dt = day.replace(hour=eh, minute=0, second=0, microsecond=0)

            slots.append(
                {
                    "start_iso": start_dt.isoformat(),
                    "end_iso": end_dt.isoformat(),
                    "label": start_dt.strftime("%A %b %d, %I:%M %p"),
                }
            )

    return slots


def create_calendar_event(
    shop: Shop,
    customer_name: str,
    phone: str,
    email: Optional[str],
    vehicle_details: str,
    slot_start_iso: str,
    slot_end_iso: str,
    estimate_summary: str,
) -> Optional[str]:
    """Create Google Calendar event if calendar_id + service are configured."""
    if not gcal_service or not shop.calendar_id:
        return None

    event_body = {
        "summary": f"Estimate - {customer_name or 'New customer'}",
        "description": (
            f"Auto-body AI estimate lead.\n\n"
            f"Name: {customer_name}\n"
            f"Phone: {phone}\n"
            f"Email: {email or 'N/A'}\n"
            f"Vehicle: {vehicle_details}\n\n"
            f"Estimate summary:\n{estimate_summary}\n"
        ),
        "start": {"dateTime": slot_start_iso},
        "end": {"dateTime": slot_end_iso},
    }

    try:
        event = gcal_service.events().insert(
            calendarId=shop.calendar_id,
            body=event_body,
        ).execute()
        return event.get("htmlLink")
    except Exception as e:
        print(f"[GCAL] Failed to create event: {e}")
        return None


# ============================================================
# LLM helpers
# ============================================================

async def call_openai_estimator(
    shop: Shop,
    user_message: str,
    image_b64: Optional[str],
) -> Dict[str, Any]:
    """
    Call OpenAI vision model with a strict JSON schema for:
    - severity
    - parts involved
    - rough cost range
    - recommended work
    """
    system_prompt = f"""
You are an expert auto body estimator working for a collision centre in Ontario, Canada called "{shop.name}".

Your job:
- Look at the vehicle photos (if provided) and the user's message.
- Classify damage severity: minor, moderate, severe.
- List affected panels/parts.
- Give a realistic Ontario repair cost range in CAD.
- Suggest what work is likely required.

Return a **single JSON object** with keys:
- severity: "minor" | "moderate" | "severe"
- affected_areas: string[]
- damage_types: string[]
- estimated_cost_min: number
- estimated_cost_max: number
- recommended_repairs: string

Do NOT include any extra text, only pure JSON.
""".strip()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    user_content: List[Dict[str, Any]] = []
    if user_message:
        user_content.append({"type": "text", "text": user_message})
    if image_b64:
        user_content.append(
            {
                "type": "input_image",
                "image": {"base64": image_b64},
            }
        )

    messages.append({"role": "user", "content": user_content})

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )

    raw = response.choices[0].message.content
    if isinstance(raw, list):
        text = "".join([c.get("text", "") for c in raw])
    else:
        text = raw or ""

    # Extract JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("Model did not return JSON")

    parsed = json.loads(m.group(0))
    return parsed


def build_estimate_sms_text(
    shop: Shop,
    severity: str,
    cost_min: float,
    cost_max: float,
    affected_areas: List[str],
    damage_types: List[str],
    recommended_repairs: str,
) -> str:
    """
    Human-friendly SMS summary that fits in 1-2 messages.
    """
    areas_str = ", ".join(affected_areas) if affected_areas else "selected areas"
    damage_str = ", ".join(damage_types) if damage_types else "body damage"

    lines = [
        f"AI Damage Estimate for {shop.name}",
        "",
        f"Severity: {severity.capitalize()}",
        f"Estimated Cost (Ontario 2025): ${int(cost_min):,} – ${int(cost_max):,}",
        f"Areas: {areas_str}",
        f"Damage Types: {damage_str}",
        "",
        "This is a visual, preliminary estimate only. Final pricing requires an in-person inspection.",
        "",
        "Would you like to book a free in-person estimate? Reply YES to see available times.",
    ]

    return "\n".join(lines)


# ============================================================
# Twilio SMS Webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Main Twilio webhook.
    - Detect shop by ?token=shop_miss_123
    - Accept text + up to 1 image
    - Return TwiML SMS response
    """
    shop = get_shop_from_request(request)

    form = await request.form()
    from_number = form.get("From") or ""
    body = form.get("Body") or ""
    num_media = int(form.get("NumMedia") or "0")

    print(f"[SMS] From {from_number} | Body: {body!r} | NumMedia={num_media}")

    normalized_from = normalize_phone(from_number)

    # Handle simple flows: YES to book, etc.
    body_lower = body.strip().lower()

    # Naive "YES" booking flow: just send them a link or short list of slots for now.
    if body_lower in {"yes", "y"}:
        slots = build_appointment_slots(shop, days_ahead=5)
        if not slots:
            reply = MessagingResponse()
            reply.message(
                "Thanks! The shop's hours are not configured yet, please call us to book an in-person estimate."
            )
            return PlainTextResponse(str(reply), media_type="text/xml")

        # Show first 5 slot labels
        preview = "\n".join(f"- {s['label']}" for s in slots[:5])
        reply = MessagingResponse()
        reply.message(
            "Great! Here are some upcoming times for an in-person estimate:\n\n"
            f"{preview}\n\n"
            "Reply with the exact day + time that works best (e.g. 'Tuesday 3pm')."
        )
        return PlainTextResponse(str(reply), media_type="text/xml")

    # If user replies with something like "Tuesday 3pm", we *could* parse and create a Calendar
    # event here. For now, we just acknowledge and tell the shop to follow up.
    if re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", body_lower):
        reply = MessagingResponse()
        reply.message(
            "Thanks! We'll lock in that time (subject to availability) and the shop will text/call you to confirm shortly."
        )
        return PlainTextResponse(str(reply), media_type="text/xml")

    # Otherwise, treat as a new estimate request
    image_b64 = None
    if num_media > 0:
        media_url = form.get("MediaUrl0")
        if media_url:
            try:
                import requests

                r = requests.get(media_url)
                r.raise_for_status()
                image_b64 = base64.b64encode(r.content).decode("utf-8")
            except Exception as e:
                print(f"[SMS] Failed to download image: {e}")

    # Call LLM estimator
    try:
        llm_json = await call_openai_estimator(shop, body, image_b64)
    except Exception as e:
        print(f"[LLM] Estimator error: {e}")
        reply = MessagingResponse()
        reply.message(
            "Sorry — our AI estimator had an issue reading that. "
            "Please send a clear photo of the damage and a short description."
        )
        return PlainTextResponse(str(reply), media_type="text/xml")

    severity = llm_json.get("severity", "moderate")
    affected_areas = llm_json.get("affected_areas", [])
    damage_types = llm_json.get("damage_types", [])
    cost_info = estimate_cost_from_llm_json(shop, severity, llm_json)

    sms_text = build_estimate_sms_text(
        shop=shop,
        severity=severity,
        cost_min=cost_info["estimated_cost_min"],
        cost_max=cost_info["estimated_cost_max"],
        affected_areas=affected_areas,
        damage_types=damage_types,
        recommended_repairs=llm_json.get("recommended_repairs", ""),
    )

    reply = MessagingResponse()
    reply.message(sms_text)

    return PlainTextResponse(str(reply), media_type="text/xml")


# ============================================================
# Simple health check
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "service": "auto-body-ai-estimator"}
