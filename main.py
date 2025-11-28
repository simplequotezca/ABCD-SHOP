import os
import json
import base64
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from zoneinfo import ZoneInfo

# ============================================================
# BASIC SETUP
# ============================================================

app = FastAPI()

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID") or ""
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN") or ""

SHOPS_JSON = os.getenv("SHOPS_JSON")
if not SHOPS_JSON:
    raise RuntimeError(
        "SHOPS_JSON env var is required. It must be a JSON list of shops."
    )

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Toronto")

# ============================================================
# MODELS & CONFIG
# ============================================================


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: Dict[str, Any] = {}


def load_shops_by_token() -> Dict[str, Shop]:
    raw = json.loads(SHOPS_JSON)
    by_token: Dict[str, Shop] = {}
    for item in raw:
        shop = Shop(**item)
        by_token[shop.webhook_token] = shop
    if not by_token:
        raise RuntimeError("SHOPS_JSON parsed but no shops found")
    return by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops_by_token()

# In-memory conversation state per phone number
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

# Lazy-loaded Google Calendar client
_calendar_service = None


def get_calendar_service():
    global _calendar_service
    if _calendar_service is None:
        if not GOOGLE_SERVICE_ACCOUNT_JSON:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON env var is required for calendar integration"
            )
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


# ============================================================
# HELPER: FORMAT TWILIO RESPONSE
# ============================================================


def twilio_xml(message: str) -> PlainTextResponse:
    """
    Wrap a message in Twilio's <Response><Message> XML so Twilio
    understands it as an SMS reply.
    """
    resp = MessagingResponse()
    resp.message(message)
    return PlainTextResponse(str(resp), media_type="application/xml")


# ============================================================
# HELPER: FETCH & ENCODE IMAGES FROM TWILIO
# ============================================================


def get_image_data_urls(form) -> List[str]:
    """
    Read all MediaUrlN items from the Twilio form,
    fetch the binary, convert to data: URLs for OpenAI Vision.
    """
    num_media = int(form.get("NumMedia", "0"))
    image_data_urls: List[str] = []

    if num_media == 0:
        return image_data_urls

    for i in range(num_media):
        url_key = f"MediaUrl{i}"
        type_key = f"MediaContentType{i}"
        media_url = form.get(url_key)
        content_type = form.get(type_key, "image/jpeg")

        if not media_url:
            continue

        logging.info(f"Fetching media from Twilio: {media_url}")

        # Twilio media URLs usually require basic auth with SID/TOKEN
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
        r = requests.get(media_url, auth=auth, timeout=20)
        r.raise_for_status()

        b64 = base64.b64encode(r.content).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64}"
        image_data_urls.append(data_url)

    logging.info(f"Prepared {len(image_data_urls)} image(s) for AI")
    return image_data_urls


# ============================================================
# HELPER: AI DAMAGE ESTIMATOR
# ============================================================


def run_damage_estimator(
    shop: Shop, image_data_urls: List[str], user_notes: str
) -> Dict[str, Any]:
    """
    Call OpenAI Vision to perform damage assessment and cost estimation.
    Returns a dict with fields like severity, cost_min, cost_max, areas, damage_types, explanation.
    """
    pricing_json = json.dumps(shop.pricing or {}, ensure_ascii=False)

    system_prompt = f"""
You are an automotive collision damage estimator for {shop.name} in Ontario, Canada.

You receive 1–3 photos of a damaged vehicle plus optional text notes from the customer.
You must:
1) Identify WHICH PANELS and AREAS are damaged (e.g., "front bumper, hood, left fender").
2) Identify the TYPES OF DAMAGE (e.g., scratch, scuff, dent, deep dent, deformation, crack, misalignment).
3) Classify SEVERITY as one of: "Minor", "Moderate", "Severe".
4) Produce a realistic COST RANGE in CAD based on this shop's pricing data:
   {pricing_json}

Rules:
- This is a VISUAL ESTIMATE ONLY, not a final repair bill.
- If you are uncertain, slightly widen the cost range rather than guessing a single number.
- If photos are very unclear, say severity is "Unclear" but still provide your best safe estimate.

Return a single JSON object with this exact schema:
{{
  "severity": "Minor | Moderate | Severe | Unclear",
  "estimated_min": 1200,
  "estimated_max": 1800,
  "currency": "CAD",
  "areas": ["front bumper", "hood"],
  "damage_types": ["paint scuff", "dent"],
  "summary": "Short 1–2 sentence customer-friendly summary.",
  "technical_notes": "1–3 bullet style sentences aimed at the technician with extra details."
}}
    """.strip()

    user_content: List[Dict[str, Any]] = []

    # Add images
    for data_url in image_data_urls:
        user_content.append(
            {
                "type": "input_image",
                "image_url": {"url": data_url, "detail": "high"},
            }
        )

    # Add text description if any
    text_notes = user_notes.strip() or "Customer did not provide extra notes."
    user_content.append({"type": "text", "text": text_notes})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = completion.choices[0].message.content
    logging.info(f"AI damage estimator raw JSON: {raw}")
    data = json.loads(raw)
    return data


# ============================================================
# HELPER: AI DATE/TIME PARSER FOR "book ..." MESSAGES
# ============================================================


def parse_booking_datetime(natural_text: str) -> datetime:
    """
    Use OpenAI to convert a natural language date/time like
    'tomorrow at 3pm' or 'Dec 1 10:30' into a timezone-aware datetime.
    """
    now = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))
    system_prompt = f"""
You are a date/time parser.

Current date and time: {now.isoformat()}
User timezone: {DEFAULT_TIMEZONE}

The user will send a phrase like:
- "tomorrow at 3pm"
- "Dec 1 at 10:30"
- "Friday 2pm"
- "next Tuesday morning"

You MUST return a JSON object with:
{{
  "iso_datetime": "YYYY-MM-DDTHH:MM:SS±HH:MM",
  "understood": true/false
}}

If you cannot confidently parse, set "understood" to false and use null for iso_datetime.
    """.strip()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_text},
        ],
    )

    raw = completion.choices[0].message.content
    logging.info(f"AI datetime parser raw JSON: {raw}")
    data = json.loads(raw)

    if not data.get("understood") or not data.get("iso_datetime"):
        raise ValueError("Could not understand requested date/time")

    dt = datetime.fromisoformat(data["iso_datetime"])
    return dt


# ============================================================
# HELPER: FIND NEXT AVAILABLE TIME SLOTS FROM CALENDAR
# ============================================================


def get_next_available_slots(shop: Shop, max_slots: int = 3) -> List[str]:
    """
    Very simple availability finder:
    - Looks 7 days ahead
    - Assumes working hours 9–5, Mon–Fri
    - Avoids hours that already have events
    Returns a list of formatted human-friendly strings.
    """
    service = get_calendar_service()
    tz = ZoneInfo(DEFAULT_TIMEZONE)
    now = datetime.now(tz)

    time_min = now.isoformat()
    time_max = (now + timedelta(days=7)).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId=shop.calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    busy_hours = set()
    for e in events:
        start = e["start"].get("dateTime")
        if not start:
            continue
        # Use hour resolution key
        busy_hours.add(start[:13])  # YYYY-MM-DDTHH

    slots: List[datetime] = []
    cursor = now

    while len(slots) < max_slots and cursor < now + timedelta(days=7):
        if cursor.weekday() >= 5:  # 5=Sat, 6=Sun
            # Jump to next Monday 9am
            days_ahead = 7 - cursor.weekday()
            cursor = (cursor + timedelta(days=days_ahead)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            continue

        # Force into business hours 9–17
        if cursor.hour < 9:
            cursor = cursor.replace(hour=9, minute=0, second=0, microsecond=0)
        if cursor.hour >= 17:
            cursor = (cursor + timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            continue

        key = cursor.isoformat()[:13]
        if key not in busy_hours:
            slots.append(cursor)

        cursor += timedelta(hours=1)

    return [dt.strftime("%a %b %d — %I:%M %p") for dt in slots]


# ============================================================
# HELPER: CREATE CALENDAR EVENT
# ============================================================


def create_booking_event(
    shop: Shop,
    when_dt: datetime,
    customer_name: str,
    phone: str,
    email: str,
    vehicle: str,
    estimate: Dict[str, Any],
):
    service = get_calendar_service()
    tz = ZoneInfo(DEFAULT_TIMEZONE)
    when_dt = when_dt.astimezone(tz)
    end_dt = when_dt + timedelta(minutes=30)

    severity = estimate.get("severity", "N/A")
    cost_min = estimate.get("estimated_min")
    cost_max = estimate.get("estimated_max")
    currency = estimate.get("currency", "CAD")

    title = f"AI Estimate – {customer_name or phone}"

    estimate_lines = [
        f"Severity: {severity}",
    ]
    if cost_min is not None and cost_max is not None:
        estimate_lines.append(
            f"Estimated Cost: {cost_min}–{cost_max} {currency}"
        )

    areas = estimate.get("areas") or []
    damage_types = estimate.get("damage_types") or []
    if areas:
        estimate_lines.append("Areas: " + ", ".join(areas))
    if damage_types:
        estimate_lines.append("Damage Types: " + ", ".join(damage_types))

    description = f"""
Customer: {customer_name or "N/A"}
Phone: {phone}
Email: {email or "N/A"}
Vehicle: {vehicle or "N/A"}

AI Estimate:
- {("\n- ").join(estimate_lines)}

Internal Estimate ID: {uuid.uuid4()}
""".strip()

    event = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": when_dt.isoformat(),
            "timeZone": DEFAULT_TIMEZONE,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": DEFAULT_TIMEZONE,
        },
    }

    created = (
        service.events()
        .insert(calendarId=shop.calendar_id, body=event)
        .execute()
    )
    logging.info(f"Created calendar event: {created.get('id')}")


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Damage Estimator is running"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str = Query(...)):
    """
    Twilio SMS/MMS webhook.
    Flow:
    1) First text (no images) -> welcome + ask for 1–3 photos.
    2) Text with 1–3 photos -> AI damage estimate + booking instructions + available slots.
    3) Text starting with "book" -> parse date/time + details, create calendar event, confirm.
    """
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=400, detail="Invalid shop token")

    form = await request.form()
    from_number = form.get("From", "")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", "0"))

    logging.info(
        f"Incoming message for shop={shop.name} from={from_number}, "
        f"body={body!r}, num_media={num_media}"
    )

    # Ensure we have a state record
    state = CONVERSATIONS.get(from_number) or {"stage": "start"}
    CONVERSATIONS[from_number] = state

    # --------------------------------------------------------
    # CASE 1: User sent images (damage photos)
    # --------------------------------------------------------
    if num_media > 0:
        try:
            image_data_urls = get_image_data_urls(form)
            if not image_data_urls:
                return twilio_xml(
                    "We could not read the photos. Please try sending the pictures again."
                )

            estimate = run_damage_estimator(shop, image_data_urls, body)

            state["stage"] = "awaiting_booking"
            state["estimate"] = estimate

            severity = estimate.get("severity", "Unclear")
            cost_min = estimate.get("estimated_min")
            cost_max = estimate.get("estimated_max")
            currency = estimate.get("currency", "CAD")
            areas = estimate.get("areas") or []
            damage_types = estimate.get("damage_types") or []
            summary = estimate.get("summary") or ""

            areas_text = ", ".join(areas) if areas else "Not clearly visible"
            types_text = ", ".join(damage_types) if damage_types else "Not clearly visible"

            # Get suggested time slots (best effort)
            try:
                slots = get_next_available_slots(shop, max_slots=3)
            except Exception as e:
                logging.exception("Error while fetching available slots")
                slots = []

            slots_text = ""
            if slots:
                slots_text = "\n\nNext available times:\n- " + "\n- ".join(slots)

            message = (
                f"Welcome to {shop.name} AI Damage Estimator.\n\n"
                f"🔍 AI Damage Assessment\n"
                f"Severity: {severity}\n"
                f"Estimated Cost: "
            )

            if cost_min is not None and cost_max is not None:
                message += f"{cost_min}–{cost_max} {currency}\n"
            else:
                message += "Not available\n"

            message += (
                f"Affected Areas: {areas_text}\n"
                f"Damage Types: {types_text}\n\n"
                f"{summary}\n\n"
                f"This is a visual, preliminary estimate only. "
                f"A technician will confirm exact repairs on site.{slots_text}\n\n"
                f"To book an appointment, reply starting with the word 'book' followed by:\n"
                f"- your preferred date & time\n"
                f"- your full name\n"
                f"- your email\n"
                f"- vehicle year/make/model\n\n"
                f"Example:\n"
                f"book tomorrow 10am; John Doe; john@email.com; 2018 Honda Civic"
            )

            return twilio_xml(message)
        except Exception as e:
            logging.exception("Error during image/AI handling")
            return twilio_xml(
                "We had trouble processing the photos. Please try again in a few minutes "
                "or send clearer pictures of the damage."
            )

    # --------------------------------------------------------
    # CASE 2: Booking message ("book ...")
    # --------------------------------------------------------
    lower_body = body.lower()
    if lower_body.startswith("book"):
        # Remove the word "book" and split extra details by ';'
        remainder = body[4:].strip()
        parts = [p.strip() for p in remainder.split(";") if p.strip()]

        preferred_text = parts[0] if len(parts) >= 1 else ""
        customer_name = parts[1] if len(parts) >= 2 else ""
        email = parts[2] if len(parts) >= 3 else ""
        vehicle = parts[3] if len(parts) >= 4 else ""

        if not preferred_text:
            return twilio_xml(
                "Please include your preferred date & time after the word 'book'.\n\n"
                "Example:\n"
                "book Friday 2pm; John Doe; john@email.com; 2018 Honda Civic"
            )

        if "estimate" not in state:
            # If somehow user skipped sending photos first
            return twilio_xml(
                "Before booking, please send 1–3 clear photos of the vehicle damage so we can prepare an estimate."
            )

        try:
            when_dt = parse_booking_datetime(preferred_text)
        except Exception:
            return twilio_xml(
                "Sorry, we couldn't understand that date/time.\n\n"
                "Please try again, for example:\n"
                "book Dec 1 at 10:30am; John Doe; john@email.com; 2018 Honda Civic"
            )

        try:
            create_booking_event(
                shop=shop,
                when_dt=when_dt,
                customer_name=customer_name,
                phone=from_number,
                email=email,
                vehicle=vehicle,
                estimate=state["estimate"],
            )
        except Exception:
            logging.exception("Error creating calendar event")
            return twilio_xml(
                "Your estimate was processed but we couldn't book the appointment in our calendar. "
                "Please call the shop directly to confirm a time."
            )

        human_time = when_dt.strftime("%A %B %d at %I:%M %p")

        message = (
            f"✅ Appointment confirmed!\n\n"
            f"We've reserved your spot at {shop.name}:\n"
            f"{human_time}\n\n"
            f"When you arrive, mention this is an AI estimate appointment linked to your phone number "
            f"{from_number}."
        )

        # Reset state after booking
        CONVERSATIONS[from_number] = {"stage": "start"}

        return twilio_xml(message)

    # --------------------------------------------------------
    # CASE 3: First contact / anything else (no images, not 'book')
    # --------------------------------------------------------
    welcome_message = (
        f"Welcome to {shop.name} AI Damage Estimator.\n\n"
        f"📸 Please send 1–3 clear photos of the vehicle damage.\n"
        f"You can also include a short message describing what happened "
        f"(e.g., 'rear ended, trunk won't close').\n\n"
        f"Our AI will analyze the photos and send you a fast, transparent repair estimate, "
        f"plus an option to book an appointment."
    )

    state["stage"] = "awaiting_photos"
    CONVERSATIONS[from_number] = state

    return twilio_xml(welcome_message)
