import os
import json
import re
import base64
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Float,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from openai import OpenAI

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dateutil import parser as date_parser
from dateutil import tz

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()

# ============================================================
# Environment / config
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

DATABASE_URL = os.getenv("DATABASE_URL")
SHOPS_JSON = os.getenv("SHOPS_JSON")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Toronto")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON is required")

if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is required for Calendar integration")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Database setup (SQLAlchemy)
# ============================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class EstimateRecord(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, index=True)  # uuid
    shop_id = Column(String, index=True)
    phone_number = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    severity = Column(String)
    cost_min = Column(Float)
    cost_max = Column(Float)
    currency = Column(String)

    image_count = Column(String)
    vehicle_info = Column(String)
    panels = Column(Text)  # JSON string for panels list

    full_text = Column(Text)  # the full human-friendly estimate text
    raw_json = Column(Text)   # raw JSON blob (AI output)


Base.metadata.create_all(bind=engine)

# ============================================================
# Shop config
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    timezone: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None


def load_shops() -> Dict[str, ShopConfig]:
    """
    SHOPS_JSON env example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123",
        "calendar_id": "shiran.bookings@gmail.com",
        "timezone": "America/Toronto"
      }
    ]
    """
    try:
        data = json.loads(SHOPS_JSON)
        by_token: Dict[str, ShopConfig] = {}
        for item in data:
            shop = ShopConfig(**item)
            by_token[shop.webhook_token] = shop
        return by_token
    except Exception as e:
        raise RuntimeError(f"Failed to parse SHOPS_JSON: {e}")


SHOPS_BY_TOKEN = load_shops()

# ============================================================
# Google Calendar client
# ============================================================

_calendar_creds = service_account.Credentials.from_service_account_info(
    json.loads(GOOGLE_SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/calendar"],
)
calendar_service = build("calendar", "v3", credentials=_calendar_creds, cache_discovery=False)

# ============================================================
# Utility helpers
# ============================================================

def get_shop_from_token(token: str) -> ShopConfig:
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404, detail="Unknown shop token")
    return shop


def get_tz(shop: ShopConfig):
    tzname = shop.timezone or DEFAULT_TIMEZONE
    return tz.gettz(tzname)


def parse_booking_details(body: str) -> Optional[Dict[str, Any]]:
    """
    Very forgiving parser. We just need:
      - name (first non-email, non-phone chunk)
      - phone (10+ digits)
      - email (@ present)
      - datetime (fuzzy parse)
    """
    text = body.strip()

    # Email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    email = email_match.group(0) if email_match else None

    # Phone: grab 10+ digits
    phone_match = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    phone = phone_match.group(0).strip() if phone_match else None

    # Name: first chunk before email/phone words
    chunks = re.split(r"[,\n;/|-]+", text)
    name = None
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if email and email in c:
            continue
        if phone and any(d.isdigit() for d in c):
            continue
        if len(c.split()) >= 1:
            name = c
            break

    # Datetime
    try:
        dt = date_parser.parse(text, fuzzy=True)
    except Exception:
        dt = None

    if not (name and (phone or email) and dt):
        return None

    return {
        "name": name,
        "phone": phone,
        "email": email,
        "datetime": dt,
        "raw": text,
    }


def create_calendar_event(shop: ShopConfig, details: Dict[str, Any], latest_estimate: Optional[EstimateRecord]) -> Dict[str, Any]:
    """
    Create a 1-hour event on the shop's calendar.
    """
    local_tz = get_tz(shop)
    start_dt = details["datetime"]
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=local_tz)
    else:
        start_dt = start_dt.astimezone(local_tz)

    end_dt = start_dt + timedelta(hours=1)

    summary = f"Auto body repair appointment - {details['name']}"
    description_lines = [
        f"Customer name: {details['name']}",
        f"Phone: {details['phone'] or 'N/A'}",
        f"Email: {details['email'] or 'N/A'}",
        "",
        "Original booking message:",
        details["raw"],
    ]
    if latest_estimate:
        description_lines.append("")
        description_lines.append(f"Estimate ID: {latest_estimate.id}")
        description_lines.append(f"Severity: {latest_estimate.severity or 'N/A'}")
        if latest_estimate.cost_min and latest_estimate.cost_max:
            description_lines.append(
                f"AI estimate: {latest_estimate.cost_min:.0f} - {latest_estimate.cost_max:.0f} {latest_estimate.currency or ''}".strip()
            )

    event_body = {
        "summary": summary,
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": str(start_dt.tzinfo),
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": str(end_dt.tzinfo),
        },
    }

    event = calendar_service.events().insert(
        calendarId=shop.calendar_id,
        body=event_body,
    ).execute()

    return event


async def fetch_twilio_media_bytes(url: str) -> bytes:
    """
    Download image bytes from Twilio, following redirects.

    This fixes the 307 Temporary Redirect error by enabling follow_redirects=True.
    """
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client_http:
        resp = await client_http.get(url, auth=auth)
        resp.raise_for_status()
        return resp.content


def build_vision_messages(shop: ShopConfig, phone: str, images_b64: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are an AI collision damage estimator for an auto body shop.\n\n"
                f"Shop name: {shop.name}\n"
                "Location: Ontario, Canada (2025 pricing).\n"
                "Analyze all images together (treat them as different angles of the SAME vehicle damage).\n\n"
                "Your job:\n"
                "1) Identify damaged panels (e.g., rear bumper cover, trunk lid, left quarter panel, tail lamps, sensors, etc.).\n"
                "2) Describe damage (scratches, dents, cracks, deformation, misalignment, gaps, etc.).\n"
                "3) Estimate realistic repair cost range in CAD, broken into labor, paint, materials, and parts.\n"
                "4) Classify severity: Minor / Moderate / Severe.\n"
                "5) Include any safety or structural concerns.\n\n"
                "Return your answer in this JSON structure ONLY (no extra commentary):\n\n"
                "{\n"
                '  \"estimate_id\": \"uuid\",\n'
                '  \"phone_number\": \"string\",\n'
                '  \"severity\": \"Minor | Moderate | Severe\",\n'
                '  \"currency\": \"CAD\",\n'
                '  \"cost_min\": number,\n'
                '  \"cost_max\": number,\n'
                '  \"labor_hours\": number,\n'
                '  \"summary\": \"short human readable summary\",\n'
                '  \"panels\": [\n'
                '    {\"panel\": \"string\", \"damage\": \"string\", \"repair_vs_replace\": \"Repair | Replace | Blend\", \"notes\": \"string\"}\n'
                "  ],\n"
                '  \"recommended_actions\": [\"string\", \"string\"],\n'
                '  \"hidden_damage_risks\": [\"string\", \"string\"],\n'
                '  \"vehicle_info\": \"string description if visible\"\n'
                "}"
            ),
        }
    ]

    for b in images_b64:
        data_url = f"data:image/jpeg;base64,{b}"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    return [
        {"role": "system", "content": "You are a professional collision damage estimator for an auto body shop."},
        {"role": "user", "content": content},
    ]


async def run_vision_estimate(shop: ShopConfig, phone: str, image_bytes_list: List[bytes]) -> Tuple[EstimateRecord, str]:
    # Convert up to 3 images to base64
    images_b64 = [
        base64.b64encode(img).decode("utf-8")
        for img in image_bytes_list[:3]
    ]

    messages = build_vision_messages(shop, phone, images_b64)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )

    raw_content = completion.choices[0].message.content.strip()

    # Try to parse JSON strictly
    try:
        data = json.loads(raw_content)
    except Exception:
        # If the model ever wraps in markdown, try to strip ```json fences
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            raise RuntimeError("AI response did not contain JSON")
        data = json.loads(match.group(0))

    estimate_id = data.get("estimate_id") or str(uuid.uuid4())
    severity = data.get("severity")
    currency = data.get("currency", "CAD")

    cost_min = float(data.get("cost_min") or 0)
    cost_max = float(data.get("cost_max") or 0)
    vehicle_info = data.get("vehicle_info") or ""
    panels = data.get("panels") or []
    summary = data.get("summary") or ""

    panels_json = json.dumps(panels)

    record = EstimateRecord(
        id=estimate_id,
        shop_id=shop.id,
        phone_number=phone,
        created_at=datetime.utcnow(),
        severity=severity,
        cost_min=cost_min,
        cost_max=cost_max,
        currency=currency,
        image_count=str(len(image_bytes_list)),
        vehicle_info=vehicle_info,
        panels=panels_json,
        full_text=summary,
        raw_json=json.dumps(data),
    )

    # Human-friendly text to send back
    panels_lines = []
    for p in panels:
        panel_name = p.get("panel", "Panel")
        dmg = p.get("damage", "")
        rvr = p.get("repair_vs_replace", "")
        note = p.get("notes", "")
        line = f"- {panel_name}: {dmg}"
        if rvr:
            line += f" ({rvr})"
        if note:
            line += f" – {note}"
        panels_lines.append(line)

    cost_line = ""
    if cost_min and cost_max:
        cost_line = f"Estimated repair range ({currency}): {cost_min:,.0f} – {cost_max:,.0f}"

    response_text = f"""AI Damage Estimate for {shop.name}

Severity: {severity or "N/A"}
{cost_line}

Visible areas involved:
{chr(10).join(panels_lines) or "• Not clearly visible from photos."}

Summary:
{summary}

Note: This is a visual, preliminary estimate only. Final pricing may change after in-person inspection.

If this looks accurate, you can book an appointment by replying with your:
- Full name
- Phone number
- Email
- Preferred date & time

Example:
John Doe, 416-555-0123, john@example.com, next Tuesday at 3pm
"""

    return record, response_text


def get_latest_estimate_for_phone(db, shop: ShopConfig, phone: str) -> Optional[EstimateRecord]:
    return (
        db.query(EstimateRecord)
        .filter(EstimateRecord.shop_id == shop.id, EstimateRecord.phone_number == phone)
        .order_by(EstimateRecord.created_at.desc())
        .first()
    )

# ============================================================
# Routes
# ============================================================

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Twilio webhook.

    Flow:
    - If message has images -> multi-image AI estimate + nice first message.
    - If text looks like booking details -> auto-write event to Google Calendar.
    - Otherwise -> prompt user to send 1–3 photos.
    """
    form = await request.form()
    from_number = (form.get("From") or "").strip()
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", "0"))

    try:
        shop = get_shop_from_token(token)
    except HTTPException:
        # Return TwiML so Twilio doesn't keep retrying
        resp = MessagingResponse()
        resp.message("Configuration error: shop not found for this number.")
        return PlainTextResponse(str(resp), media_type="application/xml", status_code=200)

    resp = MessagingResponse()

    # --------------------------------------------------------
    # 1) If there are images → run AI estimate
    # --------------------------------------------------------
    if num_media > 0:
        # Friendly first message
        resp.message(
            f"📸 Thanks! We received your photos.\n\n"
            f"Our AI estimator for {shop.name} is analyzing the damage now — "
            "you’ll receive a detailed breakdown shortly."
        )

        # Fetch all images (up to 3) with redirect handling
        image_bytes_list: List[bytes] = []
        for i in range(min(num_media, 3)):
            media_url = form.get(f"MediaUrl{i}")
            if not media_url:
                continue
            try:
                img_bytes = await fetch_twilio_media_bytes(media_url)
                image_bytes_list.append(img_bytes)
            except Exception as e:
                # If one image fails, continue with others
                print(f"Error downloading media {i}: {e}")

        if not image_bytes_list:
            resp.message(
                "⚠️ We had trouble downloading your photos. "
                "Please resend them, or make sure MMS is enabled on your phone."
            )
            return PlainTextResponse(str(resp), media_type="application/xml")

        # Run AI & store in DB
        db = SessionLocal()
        try:
            record, estimate_text = await run_vision_estimate(shop, from_number, image_bytes_list)
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"AI Processing Error: {e}")
            resp.message(
                "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
                "Please try again in a few minutes."
            )
            return PlainTextResponse(str(resp), media_type="application/xml")
        finally:
            db.close()

        resp.message(estimate_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # --------------------------------------------------------
    # 2) No images → maybe booking details
    # --------------------------------------------------------
    # Try to parse booking info
    details = parse_booking_details(body)
    if details:
        db = SessionLocal()
        try:
            latest_estimate = get_latest_estimate_for_phone(db, shop, from_number)
        finally:
            db.close()

        try:
            event = create_calendar_event(shop, details, latest_estimate)
            start_str = event["start"]["dateTime"]
            resp.message(
                "✅ Thanks! Your appointment request has been added to our calendar.\n\n"
                f"🗓 Requested time: {start_str}\n\n"
                f"{shop.name} will review and confirm with you by phone or email."
            )
        except Exception as e:
            print(f"Calendar error: {e}")
            resp.message(
                "✅ Thanks for your details.\n\n"
                "We couldn't auto-add this to the calendar, but the shop has received your info "
                "and will contact you to confirm your appointment."
            )

        return PlainTextResponse(str(resp), media_type="application/xml")

    # --------------------------------------------------------
    # 3) Fallback: no images, no booking details → onboarding message
    # --------------------------------------------------------
    resp.message(
        f"Hi! This is the AI damage estimator for {shop.name}.\n\n"
        "To get a fast, no-obligation estimate:\n"
        "• Send 1–3 clear photos of the damage\n"
        "• Include close-ups and wider angles if possible\n\n"
        "Once we receive the photos, our AI will break down the damage and estimated repair range."
    )
    return PlainTextResponse(str(resp), media_type="application/xml")
