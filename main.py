import os
import json
import re
import base64
from datetime import datetime, timedelta, timezone, date, time
from typing import Optional, Dict, Any, List, Tuple
import uuid

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from dateutil import parser as date_parser

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

# SQLAlchemy for lead storage
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Float
from sqlalchemy.orm import sessionmaker, declarative_base

from pydantic import BaseModel, EmailStr

# Timezone support
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Fallback later

# ============================================================
# VERSION IDENTIFIER
# ============================================================

VERSION = "simplequotez_full_v1_webapp"

# ============================================================
# FASTAPI APP + OPENAI SETUP
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "America/Toronto")

APP_BASE_URL = os.getenv("APP_BASE_URL")
if not APP_BASE_URL:
    raise RuntimeError("APP_BASE_URL is required (ex: https://yourapp.up.railway.app)")

# Folder for photos
PHOTOS_DIR = "photos"
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Local timezone
if ZoneInfo is not None:
    try:
        LOCAL_TZ = ZoneInfo(DEFAULT_TZ)
    except Exception:
        LOCAL_TZ = timezone.utc
else:
    LOCAL_TZ = timezone.utc

print(f"[BOOT] SimpleQuotez AI Estimator – VERSION={VERSION} TZ={DEFAULT_TZ}")

# ============================================================
# MULTI-SHOP CONFIG
# ============================================================

class ShopConfig:
    def __init__(
        self,
        id: str,
        name: str,
        webhook_token: str,
        calendar_id: Optional[str] = None,
        hours: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.name = name
        self.webhook_token = webhook_token
        self.calendar_id = calendar_id
        self.hours = hours or {}

def load_shops() -> Dict[str, ShopConfig]:
    shops_by_token = {}
    raw = os.getenv("SHOPS_JSON")

    if raw:
        try:
            data = json.loads(raw)
            for s in data:
                shops_by_token[s["webhook_token"]] = ShopConfig(
                    id=s.get("id", s["webhook_token"]),
                    name=s.get("name", "Collision Centre"),
                    webhook_token=s["webhook_token"],
                    calendar_id=s.get("calendar_id"),
                    hours=s.get("hours"),
                )
        except Exception as e:
            print("Error parsing SHOPS_JSON:", e)

    if not shops_by_token:
        print("WARNING: No SHOPS_JSON set — using a default shop.")
        shops_by_token["shop_miss_123"] = ShopConfig(
            id="miss",
            name="Mississauga Collision Centre",
            webhook_token="shop_miss_123",
            calendar_id=os.getenv("DEFAULT_CALENDAR_ID"),
        )

    return shops_by_token

SHOPS_BY_TOKEN = load_shops()
SHOPS_BY_ID: Dict[str, ShopConfig] = {s.id: s for s in SHOPS_BY_TOKEN.values()}

def get_shop_by_id(shop_id: str) -> ShopConfig:
    shop = SHOPS_BY_ID.get(shop_id)
    if not shop:
        raise HTTPException(status_code=404, detail="Unknown shop_id")
    return shop

# ============================================================
# DATABASE: LEAD STORAGE
# ============================================================

from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Float
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class Lead(Base):
    __tablename__ = "leads"

    id = Column(String, primary_key=True)  # uuid4
    shop_id = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    vehicle = Column(String, nullable=True)
    vin = Column(String, nullable=True)

    severity = Column(String, nullable=True)
    estimate_min = Column(Float, nullable=True)
    estimate_max = Column(Float, nullable=True)
    estimate_summary = Column(Text, nullable=True)
    damaged_areas = Column(Text, nullable=True)
    damage_types = Column(Text, nullable=True)

    photo_gallery_url = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(LOCAL_TZ))
    appointment_booked = Column(Boolean, default=False)


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL not set – DB disabled.")
    engine = None
    SessionLocal = None
else:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        print("Lead DB ready.")
    except Exception as e:
        print("DB init error:", e)
        engine = None
        SessionLocal = None

# ============================================================
# EMAIL (SENDGRID)
# ============================================================

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_FROM = os.getenv("FROM_EMAIL", "AI Estimator – SimpleQuotez <simplequotez@yahoo.com>")
SHOP_NOTIFICATION_EMAIL = os.getenv("SHOP_NOTIFICATION_EMAIL", "shiran.bookings@gmail.com")

def send_booking_email(shop, description: str):
    if not SENDGRID_API_KEY:
        print("SENDGRID_API_KEY missing – skipping email.")
        return
    try:
        subject = f"New AI Booking – {shop.name}"
        message = Mail(
            from_email=EMAIL_FROM,
            to_emails=SHOP_NOTIFICATION_EMAIL,
            subject=subject,
            plain_text_content=description,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print("Email sent:", response.status_code)
    except Exception as e:
        print("SendGrid error:", e)

# ============================================================
# GOOGLE CALENDAR – SERVICE ACCOUNT
# ============================================================

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
calendar_service = None

def init_calendar():
    global calendar_service
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("No Google credentials found.")
        return

    try:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=CALENDAR_SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        print("Google Calendar Ready.")
    except Exception as e:
        print("Calendar init error:", e)

init_calendar()

# ============================================================
# IMAGE DOWNLOAD + SAVE + GALLERY LINKS
# ============================================================

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

def save_customer_photo(image_bytes: bytes) -> str:
    photo_id = str(uuid.uuid4())
    file_path = os.path.join(PHOTOS_DIR, f"{photo_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return f"{APP_BASE_URL}/images/{photo_id}.jpg"

def build_gallery_url(image_urls: List[str]) -> Optional[str]:
    if not image_urls:
        return None
    payload = base64.urlsafe_b64encode(json.dumps(image_urls).encode()).decode()
    return f"{APP_BASE_URL}/photos/view/{payload}"

# ============================================================
# STATIC IMAGE / GALLERY ROUTES
# ============================================================

@app.get("/images/{photo_id}.jpg")
async def serve_image(photo_id: str):
    file_path = os.path.join(PHOTOS_DIR, f"{photo_id}.jpg")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    with open(file_path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")

@app.get("/photos/view/{payload}")
async def view_photos(payload: str):
    """
    Clean viewer for all images in an estimate.
    """
    try:
        decoded = base64.urlsafe_b64decode(payload.encode()).decode()
        urls = json.loads(decoded)
        assert isinstance(urls, list)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired gallery link")

    cards = "".join(
        f"""
        <div style="background:#111;border-radius:16px;padding:12px;
                    border:1px solid #2a2a2a;margin-bottom:18px;">
            <img src="{u}" style="width:100%;border-radius:12px;">
        </div>
        """
        for u in urls
    )

    html = f"""
    <html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>Customer Photos</title>
    </head>
    <body style="background:#000;color:#e5e7eb;font-family:system-ui;padding:20px;">
        <div style="max-width:700px;margin:0 auto;">
            <h2 style="margin-bottom:10px;">Customer Photos</h2>
            <p style="color:#9ca3af;margin-bottom:20px;">
                Uploaded through SimpleQuotez AI Estimator.
            </p>
            {cards or "<p>No images found.</p>"}
        </div>
    </body></html>
    """

    return HTMLResponse(html)

# ============================================================
# OPENAI DAMAGE ANALYSIS
# ============================================================

async def analyze_damage(image_bytes_list: List[bytes]) -> Optional[Dict[str, Any]]:
    system_prompt = (
        "You are an Ontario (2025) auto-body damage estimator.\n"
        "Return STRICT JSON ONLY.\n"
        "- Use DRIVER POV for left/right.\n"
        "- Detect structural/suspension/safety damage.\n"
        "- Severity → minor | moderate | severe.\n"
        "- Price guidelines: minor 350–1800, moderate 1800–4500, severe 4000–12000+.\n"
        "- JSON keys: severity, estimated_cost_min, estimated_cost_max,\n"
        "  damaged_areas, damage_types, summary.\n"
    )

    content = [{"type": "text", "text": "Analyze the vehicle damage from these photos."}]
    for bytes_ in image_bytes_list:
        b64 = base64.b64encode(bytes_).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=600,
            temperature=0.2
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("AI ERROR:", e)
        return None

# ============================================================
# BOOKING MESSAGE PARSING (SMS FLOW)
# ============================================================

VIN_REGEX = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

DAY_TIME_PHRASE_REGEX = re.compile(
    r"\b(?:this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight)\b[^\n,;]*",
    flags=re.IGNORECASE,
)

TIME_ONLY_REGEX = re.compile(
    r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    flags=re.IGNORECASE,
)

YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

MONTH_DATE_PHRASE_REGEX = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\.?\s*\d{1,2}(?:st|nd|rd|th)?[^\n,;]*",
    flags=re.IGNORECASE,
)

def extract_first(regex: re.Pattern, text: str):
    m = regex.search(text)
    if not m:
        return None, text
    value = m.group(1 if m.lastindex else 0).strip()
    new_text = text[:m.start()] + " " + text[m.end():]
    return value, new_text

def normalize_datetime_phrase(dt: str):
    if not dt: return dt
    dt = re.sub(r"(?i)\b(jan|feb|mar|apr|aug|sep|sept|oct|nov|dec)\.", r"\1 ", dt)
    dt = re.sub(
        r"(?i)\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s*(\d{1,2})(st|nd|rd|th)?",
        r"\1 \2",
        dt,
    )
    dt = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", dt)
    return re.sub(r"\s+", " ", dt).trim()

def parse_booking_message(body: str) -> Dict[str, Any]:
    original = body.strip()
    cleaned = re.sub(r"\bbook\b", "", original, flags=re.IGNORECASE).strip()

    vin, cleaned = extract_first(VIN_REGEX, cleaned)
    email, cleaned = extract_first(EMAIL_REGEX, cleaned)
    phone, cleaned = extract_first(PHONE_REGEX, cleaned)
    cleaned_no_years = YEAR_REGEX.sub(" ", cleaned)

    dt_source = None
    m_month = MONTH_DATE_PHRASE_REGEX.search(cleaned_no_years)
    if m_month:
        dt_source = m_month.group(0).strip()
    else:
        m_day = DAY_TIME_PHRASE_REGEX.search(cleaned_no_years)
        if m_day:
            dt_source = m_day.group(0).strip()
        else:
            m_time = TIME_ONLY_REGEX.search(cleaned_no_years)
            if m_time:
                dt_source = m_time.group(0).strip()

    preferred_dt = None
    if dt_source:
        try:
            normalized = normalize_datetime_phrase(dt_source)
            preferred_dt = date_parser.parse(
                normalized, fuzzy=True, default=datetime.now(LOCAL_TZ)
            )
            if preferred_dt.tzinfo is None:
                preferred_dt = preferred_dt.replace(tzinfo=LOCAL_TZ)
        except:
            preferred_dt = None

    leftover = cleaned_no_years.replace(dt_source or "", " ").strip()
    tokens = [t for t in re.split(r"[,\n]+|\s{2,}", leftover) if t.strip()]
    name = None
    vehicle = None
    if len(tokens) <= 3:
        name = " ".join(tokens)
    else:
        name = " ".join(tokens[:3])
        vehicle = " ".join(tokens[3:])

    return {
        "raw": original,
        "name": name,
        "phone": phone,
        "email": email,
        "vehicle": vehicle,
        "vin": vin,
        "preferred_dt": preferred_dt,
    }

  # ============================================================
# CALENDAR EVENT CREATION
# ============================================================

def create_calendar_event(
    shop: ShopConfig,
    title: str,
    description: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    if not calendar_service:
        print("Calendar not initialized — skipping event.")
        return None

    if not shop.calendar_id:
        print(f"Shop {shop.id} missing calendar_id — skipping event.")
        return None

    try:
        # Default 30-min slot if no range provided
        if start_time is None:
            start_time = datetime.now(LOCAL_TZ)
        if end_time is None:
            end_time = start_time + timedelta(minutes=30)

        # Ensure TZ
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=LOCAL_TZ)
        else:
            start_time = start_time.astimezone(LOCAL_TZ)

        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=LOCAL_TZ)
        else:
            end_time = end_time.astimezone(LOCAL_TZ)

        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_time.isoformat(), "timeZone": DEFAULT_TZ},
            "end": {"dateTime": end_time.isoformat(), "timeZone": DEFAULT_TZ},
        }

        created = calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event
        ).execute()

        print("Created Calendar Event:", created.get("id"))
        return created.get("id")

    except Exception as e:
        print("CALENDAR ERROR:", e)
        return None


# ============================================================
# LEAD DATABASE HELPERS
# ============================================================

def save_lead_to_db(
    shop: ShopConfig,
    phone: str,
    severity: Optional[str],
    est_min: Optional[float],
    est_max: Optional[float],
    summary: str,
    damaged_areas: List[str],
    damage_types: List[str],
    gallery_url: Optional[str],
):
    if not SessionLocal:
        return

    session = SessionLocal()
    try:
        lead = Lead(
            id=str(uuid.uuid4()),
            shop_id=shop.id,
            phone=phone,
            severity=severity,
            estimate_min=est_min,
            estimate_max=est_max,
            estimate_summary=summary,
            damaged_areas=", ".join(damaged_areas or []),
            damage_types=", ".join(damage_types or []),
            photo_gallery_url=gallery_url,
            created_at=datetime.now(LOCAL_TZ),
            appointment_booked=False,
        )
        session.add(lead)
        session.commit()
    except Exception as e:
        print("DB ERROR (save_lead):", e)
        session.rollback()
    finally:
        session.close()


def mark_leads_booked(shop: ShopConfig, phone: str):
    if not SessionLocal:
        return

    session = SessionLocal()
    try:
        leads = session.query(Lead).filter(
            Lead.shop_id == shop.id, Lead.phone == phone
        ).all()
        for lead in leads:
            lead.appointment_booked = True
        session.commit()
    except Exception as e:
        print("DB ERROR (mark_leads_booked):", e)
        session.rollback()
    finally:
        session.close()


# ============================================================
# LOGGING ESTIMATES & BOOKINGS
# ============================================================

def log_estimate_lead(shop: ShopConfig, from_number: str, result: Dict[str, Any]):
    severity = result.get("severity")
    est_min = result.get("estimated_cost_min")
    est_max = result.get("estimated_cost_max")
    areas = result.get("damaged_areas", []) or []
    dtypes = result.get("damage_types", []) or []
    summary = result.get("summary", "")

    hosted = (
        result.get("hosted_image_urls")
        or result.get("image_urls")
        or result.get("image_data_urls")
        or []
    )

    gallery_url = build_gallery_url(hosted) if hosted else None

    desc = [
        "AI Estimate Lead (No Booking Yet)",
        f"Shop: {shop.name}",
        f"Customer Phone: {from_number}",
        "",
        f"Severity: {severity}",
        f"Estimated Cost: ${est_min} – ${est_max}",
        f"Areas: {', '.join(areas)}",
        f"Damage Types: {', '.join(dtypes)}",
        "",
        "Summary:",
        summary,
    ]

    if gallery_url:
        desc.append("")
        desc.append("Photos:")
        desc.append(gallery_url)

    create_calendar_event(
        shop,
        title=f"Lead – {shop.name}",
        description="\n".join(desc),
    )

    save_lead_to_db(
        shop=shop,
        phone=from_number,
        severity=severity,
        est_min=est_min,
        est_max=est_max,
        summary=summary,
        damaged_areas=areas,
        damage_types=dtypes,
        gallery_url=gallery_url,
    )


def log_booking_event(
    shop: ShopConfig,
    from_number: str,
    booking: Dict[str, Any],
    last_estimate: Optional[Dict[str, Any]],
):
    name = booking.get("name") or "Unknown"
    phone = booking.get("phone") or from_number
    email = booking.get("email") or "N/A"
    vehicle = booking.get("vehicle") or "Not provided"
    vin = booking.get("vin") or "N/A"

    desc = [
        f"Booking for {shop.name}",
        f"Name: {name}",
        f"Phone: {phone}",
        f"Email: {email}",
        f"Vehicle: {vehicle}",
        f"VIN: {vin}",
        "",
        "Original SMS:",
        booking.get("raw", ""),
    ]

    gallery_url = None
    if last_estimate:
        desc.extend(
            [
                "",
                "AI Estimate:",
                f"Severity: {last_estimate.get('severity')}",
                f"Range: ${last_estimate.get('estimated_cost_min')} – ${last_estimate.get('estimated_cost_max')}",
                f"Areas: {', '.join(last_estimate.get('damaged_areas', []))}",
                f"Types: {', '.join(last_estimate.get('damage_types', []))}",
                "",
                "Summary:",
                last_estimate.get("summary", ""),
            ]
        )

        hosted = last_estimate.get("hosted_image_urls", [])
        if hosted:
            gallery_url = build_gallery_url(hosted)
            desc.extend(["", "Photos:", gallery_url])

    start_dt = booking.get("preferred_dt")
    event_id = create_calendar_event(
        shop,
        title=f"Appointment – {shop.name}",
        description="\n".join(desc),
        start_time=start_dt,
    )

    mark_leads_booked(shop, phone)
    send_booking_email(shop, "\n".join(desc))

    return event_id


# ============================================================
# MEMORY STORAGE FOR ESTIMATES (SMS + WEB)
# ============================================================

LAST_ESTIMATE_BY_PHONE: Dict[str, Dict[str, Any]] = {}
ESTIMATES_BY_ID: Dict[str, Dict[str, Any]] = {}


# ============================================================
# SMS WEBHOOK — FULL FUNCTIONALITY PRESERVED
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Handles all SMS interactions:
    - first-time message
    - photo upload → AI estimate
    - reply 1/2 flow
    - BOOK message → parsed → calendar event
    """
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        return Response("Invalid shop token.", status_code=403)

    form = await request.form()
    from_number = form.get("From") or "Unknown"
    body_raw = form.get("Body") or ""
    body = body_raw.strip()
    num_media = int(form.get("NumMedia", "0"))

    print(f"[SMS] shop={shop.id} from={from_number} media={num_media} body={body_raw!r}")

    reply = MessagingResponse()
    normalized = re.sub(r"\s+", "", body)

    # BOOKING FLOW
    if num_media == 0 and re.search(r"\bbook\b", body, flags=re.IGNORECASE):
        booking = parse_booking_message(body)
        last_est = LAST_ESTIMATE_BY_PHONE.get(from_number)

        log_booking_event(shop, from_number, booking, last_est)

        reply.message("📅 Your appointment request has been received! The shop will confirm shortly.")
        return Response(str(reply), media_type="application/xml")

    # QUICK REPLIES (no media)
    if num_media == 0:
        if normalized == "1":
            reply.message(
                "👍 Great! The shop will review your estimate.\n"
                "To book: TEXT **BOOK** + name, phone, email, car info & preferred time."
            )
            return Response(str(reply), media_type="application/xml")

        if normalized == "2":
            reply.message("📸 Please send 1–3 new photos from different angles.")
            return Response(str(reply), media_type="application/xml")

        reply.message(
            f"👋 Welcome to {shop.name}!\n"
            "Please send 1–3 clear photos of the damage for an AI estimate."
        )
        return Response(str(reply), media_type="application/xml")

    # PHOTO → AI ESTIMATE
    images = []
    hosted_urls = []
    image_urls = []

    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if not url:
            continue
        image_urls.append(url)
        try:
            img_bytes = await download_twilio_image(url)
            images.append(img_bytes)
            hosted_urls.append(save_customer_photo(img_bytes))
        except Exception as e:
            print("DOWNLOAD ERROR:", e)

    reply.message("🔍 Analyzing your photos…")
    ai_result = await analyze_damage(images)

    if not ai_result:
        reply.message("⚠️ Sorry, the AI couldn't analyze your image. Try different angles.")
        return Response(str(reply), media_type="application/xml")

    ai_result["hosted_image_urls"] = hosted_urls
    ai_result["image_urls"] = image_urls

    LAST_ESTIMATE_BY_PHONE[from_number] = ai_result
    log_estimate_lead(shop, from_number, ai_result)

    out = (
        f"🛠 {shop.name} – AI Estimate\n"
        f"Severity: {ai_result['severity']}\n"
        f"Range: ${ai_result['estimated_cost_min']} – ${ai_result['estimated_cost_max']}\n"
        f"Areas: {', '.join(ai_result['damaged_areas'])}\n"
        "Reply 1 if accurate, or 2 to send new photos.\n"
        "To book: TEXT **BOOK** + your name, phone, email, car, and preferred time."
    )

    reply.message(out)
    return Response(str(reply), media_type="application/xml")

  # ============================================================
# WEB API MODELS (FOR CLICKABLE LINK FRONTEND)
# ============================================================

from pydantic import BaseModel, EmailStr

class EstimateResponse(BaseModel):
    estimate_id: str
    shop_id: str
    severity: str
    estimate_min: float
    estimate_max: float
    currency: str = "CAD"
    areas: List[str]
    damage_types: List[str]
    summary: str

class TimeSlot(BaseModel):
    start: datetime
    end: datetime

class AvailabilityRequest(BaseModel):
    shop_id: str
    date: date

class AvailabilityResponse(BaseModel):
    shop_id: str
    date: date
    slots: List[TimeSlot]

class BookingRequest(BaseModel):
    shop_id: str
    estimate_id: str
    customer_name: str
    phone: str
    email: EmailStr
    vehicle_year: Optional[int] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    slot_start: datetime
    slot_end: datetime

class BookingResponse(BaseModel):
    success: bool
    message: str
    calendar_event_id: Optional[str] = None


# ============================================================
# AVAILABILITY / SHOP HOURS LOGIC
# ============================================================

WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

def get_shop_hours_for_date(shop: ShopConfig, target_date: date):
    """
    Returns (start_dt, end_dt) for the shop's working hours on a specific date.
    If closed → return None.
    """

    day_name = WEEKDAY_NAMES[target_date.weekday()]
    hours_cfg = getattr(shop, "hours", {}) or {}
    day_cfg = hours_cfg.get(day_name)

    # If hours are defined in SHOPS_JSON
    if day_cfg:
        if day_cfg.get("closed"):
            return None
        try:
            open_h, open_m = map(int, day_cfg.get("open", "09:00").split(":"))
            close_h, close_m = map(int, day_cfg.get("close", "17:00").split(":"))
        except:
            open_h, open_m = 9, 0
            close_h, close_m = 17, 0
    else:
        # Default fallback
        if day_name == "saturday":
            open_h, open_m = 10, 0
            close_h, close_m = 14, 0
        elif day_name == "sunday":
            return None
        else:
            open_h, open_m = 9, 0
            close_h, close_m = 17, 0

    start_dt = datetime.combine(target_date, time(open_h, open_m, tzinfo=LOCAL_TZ))
    end_dt = datetime.combine(target_date, time(close_h, close_m, tzinfo=LOCAL_TZ))
    return start_dt, end_dt


def get_busy_intervals(shop: ShopConfig, start_dt: datetime, end_dt: datetime):
    """
    Fetch shop's busy times using Google Calendar freebusy().
    Returns a list of (busy_start, busy_end).
    """
    if not calendar_service or not shop.calendar_id:
        return []

    try:
        body = {
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "timeZone": DEFAULT_TZ,
            "items": [{"id": shop.calendar_id}],
        }

        resp = calendar_service.freebusy().query(body=body).execute()
        busy = resp["calendars"][shop.calendar_id].get("busy", [])
        intervals = []
        for b in busy:
            b_start = date_parser.parse(b["start"]).astimezone(LOCAL_TZ)
            b_end = date_parser.parse(b["end"]).astimezone(LOCAL_TZ)
            intervals.append((b_start, b_end))
        return intervals

    except Exception as e:
        print("FREEBUSY ERROR:", e)
        return []


def get_availability_for_shop(shop_id: str, target_date: date) -> List[TimeSlot]:
    """
    Generates all 30-minute available time slots for the date.
    """
    shop = get_shop_by_id(shop_id)
    hours = get_shop_hours_for_date(shop, target_date)
    if not hours:
        return []

    day_start, day_end = hours
    busy_intervals = get_busy_intervals(shop, day_start, day_end)

    slots = []
    slot_length = timedelta(minutes=30)
    cursor = day_start

    while cursor + slot_length <= day_end:
        slot_start = cursor
        slot_end = cursor + slot_length

        overlapped = any(
            not (slot_end <= b_start or slot_start >= b_end)
            for b_start, b_end in busy_intervals
        )

        if not overlapped:
            slots.append(TimeSlot(start=slot_start, end=slot_end))

        cursor += slot_length

    return slots


# ============================================================
# API: /api/estimate  (UPLOAD → AI ESTIMATE)
# ============================================================

@app.post("/api/estimate", response_model=EstimateResponse)
async def api_estimate(shop_id: str = Form(...), images: List[UploadFile] = File(...)):

    shop = get_shop_by_id(shop_id)

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed.")

    image_bytes_list = []
    hosted_urls = []

    for img in images:
        data = await img.read()
        if not data:
            continue
        image_bytes_list.append(data)
        hosted_urls.append(save_customer_photo(data))

    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="Could not process images.")

    ai = await analyze_damage(image_bytes_list)
    if ai is None:
        raise HTTPException(status_code=500, detail="AI could not process images.")

    estimate_id = str(uuid.uuid4())
    ai["estimate_id"] = estimate_id
    ai["hosted_image_urls"] = hosted_urls

    # Store for later booking
    ESTIMATES_BY_ID[estimate_id] = ai

    # Log as a lead with pseudo phone
    log_estimate_lead(shop, "WEB_LEAD", ai)

    return EstimateResponse(
        estimate_id=estimate_id,
        shop_id=shop.id,
        severity=ai["severity"],
        estimate_min=float(ai["estimated_cost_min"]),
        estimate_max=float(ai["estimated_cost_max"]),
        areas=ai.get("damaged_areas", []),
        damage_types=ai.get("damage_types", []),
        summary=ai.get("summary", ""),
    )


# ============================================================
# API: /api/availability  (RETURN FREE TIME SLOTS)
# ============================================================

@app.post("/api/availability", response_model=AvailabilityResponse)
def api_availability(payload: AvailabilityRequest):
    slots = get_availability_for_shop(payload.shop_id, payload.date)
    return AvailabilityResponse(
        shop_id=payload.shop_id,
        date=payload.date,
        slots=slots,
    )

  # ============================================================
# API: /api/book  (WEB BOOKING ENDPOINT)
# ============================================================

@app.post("/api/book", response_model=BookingResponse)
async def api_book(payload: BookingRequest):
    """
    Final step for the web flow:
    - Validates estimate_id
    - Creates Calendar event
    - Marks DB leads as booked
    - Sends email notification
    """
    shop = get_shop_by_id(payload.shop_id)

    est = ESTIMATES_BY_ID.get(payload.estimate_id)
    if not est:
        raise HTTPException(status_code=404, detail="estimate_id not found or expired")

    # Normalize times to LOCAL_TZ
    start_dt = payload.slot_start
    end_dt = payload.slot_end

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=LOCAL_TZ)
    else:
        start_dt = start_dt.astimezone(LOCAL_TZ)

    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=LOCAL_TZ)
    else:
        end_dt = end_dt.astimezone(LOCAL_TZ)

    # Estimate details
    severity = est.get("severity", "unknown")
    est_min = est.get("estimated_cost_min")
    est_max = est.get("estimated_cost_max")
    areas = est.get("damaged_areas", []) or []
    dtypes = est.get("damage_types", []) or []
    summary = est.get("summary", "")

    hosted = (
        est.get("hosted_image_urls")
        or est.get("image_urls")
        or est.get("image_data_urls")
        or []
    )
    gallery_url = build_gallery_url(hosted) if hosted else None

    # Build event description
    lines = [
        f"WEB BOOKING – {shop.name}",
        "",
        "Customer",
        f"Name: {payload.customer_name}",
        f"Phone: {payload.phone}",
        f"Email: {payload.email}",
        "",
        "Vehicle",
        f"Year: {payload.vehicle_year or 'N/A'}",
        f"Make: {payload.vehicle_make or 'N/A'}",
        f"Model: {payload.vehicle_model or 'N/A'}",
        "",
        "AI Estimate",
        f"Severity: {severity}",
        f"Estimated Range: ${est_min} – ${est_max}",
        f"Areas: {', '.join(areas) or 'N/A'}",
        f"Damage Types: {', '.join(dtypes) or 'N/A'}",
        "",
        "Summary:",
        summary or "N/A",
    ]

    if gallery_url:
        lines.extend(["", "Photos:", gallery_url])

    description = "\n".join(lines)

    event_id = create_calendar_event(
        shop=shop,
        title=f"Appointment – {shop.name}",
        description=description,
        start_time=start_dt,
        end_time=end_dt,
    )

    if not event_id:
        # Still mark as booked in DB / log email, but report partial failure
        print("WARNING: Calendar event not created for web booking.")

    # Mark as booked in DB (if DB configured)
    mark_leads_booked(shop, payload.phone)

    # Email notification
    send_booking_email(shop, description)

    return BookingResponse(
        success=True,
        message="Booking request submitted to the shop.",
        calendar_event_id=event_id,
    )

  # ============================================================
# SIMPLEQUOTEZ WEB APP – /quote
# ============================================================

@app.get("/quote")
async def quote_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>SimpleQuotez – AI Damage Estimator</title>
      <style>
        :root {
          --sq-bg: #000000;
          --sq-surface: #111111;
          --sq-surface-soft: #151515;
          --sq-border: #27272a;
          --sq-text: #e5e7eb;
          --sq-text-muted: #9ca3af;
          --sq-accent: #3b82f6;
          --sq-accent-soft: rgba(59,130,246,.14);
          --sq-danger: #ef4444;
          --sq-radius-lg: 18px;
          --sq-radius-pill: 999px;
          --sq-shadow-soft: 0 22px 45px rgba(0,0,0,.7);
        }
        * { box-sizing:border-box; }
        body {
          margin:0;
          font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          background:radial-gradient(circle at top,#111827 0,#000 55%,#000 100%);
          color:var(--sq-text);
          min-height:100vh;
          display:flex;
          justify-content:center;
        }
        .sq-page { width:100%; max-width:520px; margin:0 auto; padding:20px 14px 32px; }
        @media(min-width:640px){ .sq-page{ padding:32px 0 40px; } }
        .sq-card{
          background:linear-gradient(140deg,#050816,#111);
          border-radius:28px;
          box-shadow:var(--sq-shadow-soft);
          border:1px solid rgba(148,163,184,.16);
          padding:22px 18px 24px;
        }
        @media(min-width:640px){ .sq-card{ padding:26px 24px 26px; } }
        .sq-header{ display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; }
        .sq-logo{ display:flex; align-items:center; gap:10px; }
        .sq-logo-mark{
          width:32px;height:32px;border-radius:999px;
          border:1px solid rgba(156,163,175,.7);
          display:flex;align-items:center;justify-content:center;
          background:radial-gradient(circle at 30% 0,rgba(59,130,246,.35),#020617);
          font-size:18px;color:var(--sq-accent);
        }
        .sq-logo-main{ font-size:14px; letter-spacing:.16em; text-transform:uppercase; color:var(--sq-text-muted); }
        .sq-logo-sub{ font-size:12px; color:rgba(148,163,184,.9); }
        .sq-step-indicator{ display:flex;align-items:center;gap:4px; }
        .sq-step-dot{ width:8px;height:8px;border-radius:999px;background:#27272f; }
        .sq-step-dot.active{ background:var(--sq-accent); box-shadow:0 0 0 4px rgba(59,130,246,.35); }
        h1{ margin:0 0 6px; font-size:20px; letter-spacing:.04em; }
        .sq-subtitle{ margin:0 0 16px; font-size:13px; color:var(--sq-text-muted); }
        .sq-section{ display:none; animation:fadeIn .24s ease-out; }
        .sq-section.active{ display:block; }
        @keyframes fadeIn{ from{opacity:0;transform:translateY(4px);} to{opacity:1;transform:translateY(0);} }
        .sq-upload-area{
          border-radius:var(--sq-radius-lg);
          border:1px dashed rgba(148,163,184,.45);
          background:radial-gradient(circle at top,rgba(59,130,246,.08),#020617);
          padding:18px 14px 16px;
          text-align:center;
          cursor:pointer;
          transition:border .15s,background .15s,transform .1s;
        }
        .sq-upload-area:hover{
          border-color:rgba(59,130,246,.85);
          background:radial-gradient(circle at top,rgba(59,130,246,.18),#020617);
          transform:translateY(-1px);
        }
        .sq-upload-icon{
          width:40px;height:40px;border-radius:999px;
          display:flex;align-items:center;justify-content:center;
          background:radial-gradient(circle at 30% 0,rgba(59,130,246,.55),#020617);
          margin:0 auto 10px;font-size:20px;
        }
        .sq-upload-title{ font-size:14px;margin-bottom:4px; }
        .sq-upload-help{ font-size:12px;color:var(--sq-text-muted);margin-bottom:4px; }
        .sq-upload-hint{ font-size:11px;color:var(--sq-text-muted); }
        .sq-file-preview{ display:flex;gap:8px;margin-top:10px;flex-wrap:wrap; }
        .sq-thumb{ width:62px;height:62px;border-radius:16px;overflow:hidden;border:1px solid rgba(148,163,184,.3);background:#020617; }
        .sq-thumb img{ width:100%;height:100%;object-fit:cover;display:block; }
        .sq-button-row{ display:flex;gap:10px;margin-top:16px; }
        .sq-btn{
          border:none;border-radius:var(--sq-radius-pill);
          padding:9px 14px;font-size:13px;font-weight:500;
          cursor:pointer;display:inline-flex;align-items:center;justify-content:center;
          transition:background .15s,transform .1s,box-shadow .15s;white-space:nowrap;
        }
        .sq-btn-primary{
          flex:1;background:linear-gradient(135deg,var(--sq-accent),#2563eb);
          color:#fff;box-shadow:0 10px 25px rgba(37,99,235,.48);
        }
        .sq-btn-primary:hover{ background:linear-gradient(135deg,#4f46e5,#2563eb);transform:translateY(-1px);box-shadow:0 14px 35px rgba(37,99,235,.7); }
        .sq-btn-secondary{
          background:#020617;border:1px solid rgba(148,163,184,.55);
          color:var(--sq-text-muted);padding-inline:13px;
        }
        .sq-btn-secondary:hover{ border-color:rgba(148,163,184,.9);transform:translateY(-1px); }
        .sq-btn[disabled]{ opacity:.4;cursor:not-allowed;box-shadow:none;transform:none; }
        .sq-pill{
          display:inline-flex;align-items:center;gap:6px;border-radius:999px;
          padding:4px 10px;font-size:11px;
          background:rgba(15,23,42,.96);
          border:1px solid rgba(148,163,184,.4);
          color:var(--sq-text-muted);
        }
        .sq-pill-dot{ width:7px;height:7px;border-radius:999px;background:currentColor; }
        .sq-pill strong{ color:var(--sq-text);font-weight:500; }
        .severity-minor{ border-color:#22c55e;color:#bbf7d0; }
        .severity-moderate{ border-color:#eab308;color:#facc15; }
        .severity-severe{ border-color:#ef4444;color:#fecaca; }
        .sq-meta-row{ display:flex;flex-wrap:wrap;gap:8px;margin-top:10px; }
        .sq-meta-chip{
          border-radius:999px;padding:4px 10px;font-size:11px;
          background:rgba(15,23,42,.96);
          border:1px solid rgba(148,163,184,.35);
          color:var(--sq-text-muted);
        }
        .sq-estimate-summary{
          margin-top:14px;padding:12px 11px;border-radius:16px;
          background:rgba(15,23,42,.96);
          border:1px solid rgba(148,163,184,.3);
          font-size:12px;color:var(--sq-text-muted);line-height:1.5;
        }
        .sq-estimate-summary strong{ color:var(--sq-text); }
        .sq-label{ font-size:12px;color:var(--sq-text-muted);margin-bottom:4px; }
        .sq-input{
          width:100%;border-radius:12px;padding:9px 10px;
          border:1px solid rgba(55,65,81,.9);
          background:#020617;color:var(--sq-text);
          font-size:13px;outline:none;
          transition:border .12s,box-shadow .12s,background .12s;
        }
        .sq-input::placeholder{ color:rgba(148,163,184,.75); }
        .sq-input:focus{
          border-color:var(--sq-accent);
          box-shadow:0 0 0 1px rgba(59,130,246,.8);
          background:#020617;
        }
        .sq-grid-2{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px; }
        .sq-field{ margin-bottom:10px; }
        .sq-calendar-shell{
          border-radius:var(--sq-radius-lg);
          border:1px solid rgba(148,163,184,.4);
          background:radial-gradient(circle at top,rgba(59,130,246,.14),#020617);
          padding:10px 10px 12px;
        }
        .sq-calendar-header{ display:flex;justify-content:space-between;align-items:center;margin-bottom:6px; }
        .sq-calendar-title{ font-size:13px;font-weight:500; }
        .sq-calendar-sub{ font-size:11px;color:var(--sq-text-muted); }
        .sq-calendar-nav{ display:inline-flex;gap:4px; }
        .sq-nav-btn{
          width:24px;height:24px;border-radius:999px;
          border:1px solid rgba(148,163,184,.55);
          background:rgba(15,23,42,.96);
          color:var(--sq-text-muted);font-size:14px;
          display:flex;align-items:center;justify-content:center;
          cursor:pointer;transition:background .12s,border .12s,transform .08s;
        }
        .sq-nav-btn:hover{ background:#020617;border-color:var(--sq-accent);transform:translateY(-.5px); }
        .sq-slot-groups{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px; }
        @media(min-width:480px){ .sq-slot-groups{ grid-template-columns:repeat(3,minmax(0,1fr)); } }
        .sq-slot-group{
          background:rgba(15,23,42,.98);
          border-radius:14px;
          padding:8px 8px 6px;
          border:1px solid rgba(148,163,184,.3);
        }
        .sq-slot-group-title{
          font-size:11px;text-transform:uppercase;
          letter-spacing:.14em;color:var(--sq-text-muted);
          margin-bottom:4px;
        }
        .sq-slots{ display:flex;flex-wrap:wrap;gap:4px; }
        .sq-slot{
          border-radius:999px;
          border:1px solid rgba(148,163,184,.5);
          padding:4px 9px;
          font-size:11px;color:var(--sq-text-muted);
          background:#020617;
          cursor:pointer;
          transition:background .12s,border .12s,color .12s,transform .08s;
        }
        .sq-slot:hover{
          border-color:var(--sq-accent);
          color:var(--sq-text);transform:translateY(-.5px);
        }
        .sq-slot.selected{
          background:var(--sq-accent);
          border-color:var(--sq-accent);
          color:#fff;
          box-shadow:0 0 0 1px rgba(59,130,246,.9);
        }
        .sq-slot-empty{ font-size:11px;color:var(--sq-text-muted);margin-top:4px; }
        .sq-footer-note{
          margin-top:12px;font-size:10px;color:var(--sq-text-muted);text-align:center;
        }
        .sq-footer-note span{ color:rgba(148,163,184,.9); }
        .sq-alert{
          margin-top:10px;border-radius:10px;padding:8px 10px;
          font-size:11px;background:rgba(248,113,113,.07);
          color:#fecaca;border:1px solid rgba(248,113,113,.55);
          display:none;
        }
        .sq-alert.show{ display:block; }
        .sq-tagline{ margin-top:10px;font-size:11px;color:var(--sq-text-muted);text-align:center; }
      </style>
    </head>
    <body>
      <div class="sq-page">
        <div class="sq-card">
          <header class="sq-header">
            <div class="sq-logo">
              <div class="sq-logo-mark">🧠</div>
              <div>
                <div class="sq-logo-main">SIMPLEQUOTEZ</div>
                <div class="sq-logo-sub">AI Damage Estimator</div>
              </div>
            </div>
            <div class="sq-step-indicator">
              <div class="sq-step-dot active" data-step-dot="1"></div>
              <div class="sq-step-dot" data-step-dot="2"></div>
              <div class="sq-step-dot" data-step-dot="3"></div>
              <div class="sq-step-dot" data-step-dot="4"></div>
              <div class="sq-step-dot" data-step-dot="5"></div>
            </div>
          </header>

          <main>
            <!-- STEP 1: Upload -->
            <section class="sq-section active" data-step="1">
              <h1>Get a fast repair estimate</h1>
              <p class="sq-subtitle">
                Upload 1–3 clear photos of the damage. Our AI will scan your vehicle and estimate the repair range.
              </p>

              <label class="sq-upload-area" id="sq-upload-area">
                <input type="file" id="sq-file-input" accept="image/*" multiple style="display:none" />
                <div class="sq-upload-icon">↑</div>
                <div class="sq-upload-title">Drop photos here or tap to select</div>
                <div class="sq-upload-help">Front, angled, and close-up shots work best.</div>
                <div class="sq-upload-hint">Up to 3 photos · JPG or PNG</div>
                <div class="sq-file-preview" id="sq-file-preview"></div>
              </label>

              <div class="sq-button-row">
                <button class="sq-btn sq-btn-secondary" type="button" id="sq-demo-btn">Try a demo</button>
                <button class="sq-btn sq-btn-primary" type="button" id="sq-estimate-btn">Get AI estimate</button>
              </div>

              <div class="sq-tagline">No downloads · No account · Estimate in under a minute.</div>
              <div class="sq-alert" id="sq-alert-upload"></div>
            </section>

            <!-- STEP 2: Estimate -->
            <section class="sq-section" data-step="2">
              <h1>Your AI estimate</h1>
              <p class="sq-subtitle">
                This is a preliminary visual estimate. Final pricing is confirmed after a short in-person inspection.
              </p>

              <div class="sq-pill" id="sq-severity-pill">
                <span class="sq-pill-dot"></span>
                <strong>Severity:</strong>
                <span id="sq-severity-text">–</span>
              </div>

              <div class="sq-meta-row">
                <div class="sq-meta-chip">
                  Est. range: <strong id="sq-price-range">$0 – $0</strong>
                </div>
                <div class="sq-meta-chip">
                  Areas: <span id="sq-areas-text">–</span>
                </div>
              </div>

              <div class="sq-estimate-summary">
                <strong>Summary</strong>
                <div id="sq-summary-text" style="margin-top:4px;">–</div>
              </div>

              <div class="sq-button-row" style="margin-top:14px;">
                <button class="sq-btn sq-btn-secondary" type="button" data-back-step="1">Back</button>
                <button class="sq-btn sq-btn-primary" type="button" id="sq-to-info-btn">Continue to booking</button>
              </div>

              <div class="sq-footer-note">
                You’ll only confirm the visit after the shop reviews your photos and estimate.
              </div>

              <div class="sq-alert" id="sq-alert-estimate"></div>
            </section>

            <!-- STEP 3: Customer info -->
            <section class="sq-section" data-step="3">
              <h1>Tell us about you</h1>
              <p class="sq-subtitle">Your details help the shop confirm your visit and send reminders.</p>

              <div class="sq-field">
                <div class="sq-label">Full name</div>
                <input class="sq-input" id="sq-name" type="text" placeholder="Jane Doe" />
              </div>

              <div class="sq-grid-2">
                <div class="sq-field">
                  <div class="sq-label">Phone number</div>
                  <input class="sq-input" id="sq-phone" type="tel" placeholder="(555) 123-4567" />
                </div>
                <div class="sq-field">
                  <div class="sq-label">Email</div>
                  <input class="sq-input" id="sq-email" type="email" placeholder="you@example.com" />
                </div>
              </div>

              <div class="sq-grid-2">
                <div class="sq-field">
                  <div class="sq-label">Vehicle year</div>
                  <input class="sq-input" id="sq-year" type="number" placeholder="2021" />
                </div>
                <div class="sq-field">
                  <div class="sq-label">Make & model</div>
                  <input class="sq-input" id="sq-make-model" type="text" placeholder="Toyota Camry" />
                </div>
              </div>

              <div class="sq-button-row" style="margin-top:16px;">
                <button class="sq-btn sq-btn-secondary" type="button" data-back-step="2">Back</button>
                <button class="sq-btn sq-btn-primary" type="button" id="sq-to-calendar-btn">Choose date & time</button>
              </div>

              <div class="sq-alert" id="sq-alert-info"></div>
            </section>

            <!-- STEP 4: Calendar -->
            <section class="sq-section" data-step="4">
              <h1>Select a time</h1>
              <p class="sq-subtitle">Only open slots for this shop are shown. Times are local to the shop.</p>

              <div class="sq-calendar-shell">
                <div class="sq-calendar-header">
                  <div>
                    <div class="sq-calendar-title" id="sq-calendar-title">–</div>
                    <div class="sq-calendar-sub">30-minute visit · Visual estimate & drop-off planning</div>
                  </div>
                  <div class="sq-calendar-nav">
                    <button class="sq-nav-btn" type="button" id="sq-prev-day">&lt;</button>
                    <button class="sq-nav-btn" type="button" id="sq-next-day">&gt;</button>
                  </div>
                </div>

                <div class="sq-slot-groups" id="sq-slot-groups"></div>
                <div class="sq-slot-empty" id="sq-slot-empty" style="display:none;">
                  No open slots for this day. Try another date.
                </div>
              </div>

              <div class="sq-button-row" style="margin-top:14px;">
                <button class="sq-btn sq-btn-secondary" type="button" data-back-step="3">Back</button>
                <button class="sq-btn sq-btn-primary" type="button" id="sq-book-btn">Confirm appointment</button>
              </div>

              <div class="sq-alert" id="sq-alert-calendar"></div>
            </section>

            <!-- STEP 5: Confirmation -->
            <section class="sq-section" data-step="5">
              <h1>You’re all set</h1>
              <p class="sq-subtitle" id="sq-confirm-text">
                Your appointment request has been sent to the shop. They’ll confirm the details shortly.
              </p>

              <div class="sq-estimate-summary">
                <strong>What happens next?</strong>
                <ul style="margin:8px 0 0;padding-left:18px;font-size:12px;color:var(--sq-text-muted);line-height:1.5;">
                  <li>The shop receives your photos, AI estimate, and preferred time.</li>
                  <li>They add your visit to their schedule and may contact you to confirm.</li>
                  <li>Final pricing is confirmed after an in-person inspection.</li>
                </ul>
              </div>

              <div class="sq-button-row" style="margin-top:16px;">
                <button class="sq-btn sq-btn-primary" type="button" id="sq-new-request-btn">Start a new request</button>
              </div>

              <div class="sq-footer-note">
                Powered by <span>SimpleQuotez</span> · Smart AI estimates for modern collision centres.
              </div>
            </section>
          </main>
        </div>
      </div>

      <script>
      (function(){
        const sections = document.querySelectorAll(".sq-section");
        const dots = document.querySelectorAll("[data-step-dot]");
        const uploadArea = document.getElementById("sq-upload-area");
        const fileInput = document.getElementById("sq-file-input");
        const filePreview = document.getElementById("sq-file-preview");
        const estimateBtn = document.getElementById("sq-estimate-btn");
        const demoBtn = document.getElementById("sq-demo-btn");
        const alertUpload = document.getElementById("sq-alert-upload");
        const alertEstimate = document.getElementById("sq-alert-estimate");
        const alertInfo = document.getElementById("sq-alert-info");
        const alertCalendar = document.getElementById("sq-alert-calendar");

        const severityPill = document.getElementById("sq-severity-pill");
        const severityText = document.getElementById("sq-severity-text");
        const priceRange = document.getElementById("sq-price-range");
        const areasText = document.getElementById("sq-areas-text");
        const summaryText = document.getElementById("sq-summary-text");
        const toInfoBtn = document.getElementById("sq-to-info-btn");

        const nameInput = document.getElementById("sq-name");
        const phoneInput = document.getElementById("sq-phone");
        const emailInput = document.getElementById("sq-email");
        const yearInput = document.getElementById("sq-year");
        const makeModelInput = document.getElementById("sq-make-model");
        const toCalendarBtn = document.getElementById("sq-to-calendar-btn");

        const calendarTitle = document.getElementById("sq-calendar-title");
        const slotGroupsContainer = document.getElementById("sq-slot-groups");
        const slotEmpty = document.getElementById("sq-slot-empty");
        const prevDayBtn = document.getElementById("sq-prev-day");
        const nextDayBtn = document.getElementById("sq-next-day");
        const bookBtn = document.getElementById("sq-book-btn");

        const confirmText = document.getElementById("sq-confirm-text");
        const newRequestBtn = document.getElementById("sq-new-request-btn");

        let currentStep = 1;
        let files = [];
        let estimateId = null;
        let currentDate = new Date();
        let selectedSlot = null;

        const params = new URLSearchParams(window.location.search);
        const shopId = params.get("shop_id") || "miss";

        function showStep(step){
          currentStep = step;
          sections.forEach(sec => {
            const s = parseInt(sec.getAttribute("data-step"),10);
            sec.classList.toggle("active", s === step);
          });
          dots.forEach(dot => {
            const s = parseInt(dot.getAttribute("data-step-dot"),10);
            dot.classList.toggle("active", s <= step);
          });
          window.scrollTo({top:0,behavior:"smooth"});
        }

        function showAlert(el,msg){
          if(!el) return;
          el.textContent = msg;
          el.classList.add("show");
          setTimeout(()=>el.classList.remove("show"),4500);
        }

        function setLoading(btn,isLoading,label){
          if(!btn) return;
          if(!btn.dataset.originalText){
            btn.dataset.originalText = btn.textContent;
          }
          if(isLoading){
            btn.disabled = true;
            btn.textContent = label || "Working…";
          }else{
            btn.disabled = false;
            btn.textContent = btn.dataset.originalText;
          }
        }

        function handleFiles(list){
          const arr = Array.from(list || []);
          if(!arr.length) return;
          if(arr.length > 3){
            showAlert(alertUpload,"Please select up to 3 photos.");
          }
          files = arr.slice(0,3);
          filePreview.innerHTML = "";
          files.forEach(file=>{
            const reader = new FileReader();
            reader.onload = ev=>{
              const d = document.createElement("div");
              d.className = "sq-thumb";
              const img = document.createElement("img");
              img.src = ev.target.result;
              d.appendChild(img);
              filePreview.appendChild(d);
            };
            reader.readAsDataURL(file);
          });
        }

        uploadArea.addEventListener("click",()=>fileInput.click());
        uploadArea.addEventListener("dragover",e=>{e.preventDefault();});
        uploadArea.addEventListener("drop",e=>{
          e.preventDefault();
          if(e.dataTransfer && e.dataTransfer.files){
            handleFiles(e.dataTransfer.files);
          }
        });
        fileInput.addEventListener("change",e=>handleFiles(e.target.files));

        document.querySelectorAll("[data-back-step]").forEach(btn=>{
          btn.addEventListener("click",()=>{
            const target = parseInt(btn.getAttribute("data-back-step"),10);
            showStep(target);
          });
        });

        demoBtn.addEventListener("click",()=>{
          estimateId = "demo-"+Date.now();
          severityText.textContent = "moderate (demo)";
          priceRange.textContent = "$1,800 – $3,200";
          areasText.textContent = "rear bumper, left quarter panel";
          summaryText.textContent =
            "From your photos it appears there is moderate impact to the rear bumper and left rear quarter panel. " +
            "This estimate assumes no hidden structural or suspension damage. Final pricing will be confirmed after an in-person inspection.";
          severityPill.classList.remove("severity-minor","severity-moderate","severity-severe");
          severityPill.classList.add("severity-moderate");
          showStep(2);
        });

        estimateBtn.addEventListener("click",async()=>{
          if(!files.length){
            showAlert(alertUpload,"Please upload at least one photo or use the demo option.");
            return;
          }
          setLoading(estimateBtn,true,"Analyzing…");
          try{
            const fd = new FormData();
            fd.append("shop_id",shopId);
            files.forEach(f=>fd.append("images",f));

            const resp = await fetch("/api/estimate",{method:"POST",body:fd});
            if(!resp.ok) throw new Error("Estimate failed");
            const data = await resp.json();

            estimateId = data.estimate_id;
            severityText.textContent = data.severity || "unknown";
            priceRange.textContent = "$"+Math.round(data.estimate_min)+" – $"+Math.round(data.estimate_max);
            areasText.textContent = (data.areas || []).join(", ") || "N/A";
            summaryText.textContent = data.summary || "Estimate generated.";

            severityPill.classList.remove("severity-minor","severity-moderate","severity-severe");
            if(data.severity === "minor") severityPill.classList.add("severity-minor");
            else if(data.severity === "severe") severityPill.classList.add("severity-severe");
            else severityPill.classList.add("severity-moderate");

            showStep(2);
          }catch(e){
            console.error(e);
            showAlert(alertUpload,"There was a problem generating your estimate. Please try again.");
          }finally{
            setLoading(estimateBtn,false);
          }
        });

        toInfoBtn.addEventListener("click",()=>{
          if(!estimateId){
            showAlert(alertEstimate,"Please generate an estimate first.");
            return;
          }
          showStep(3);
        });

        toCalendarBtn.addEventListener("click",()=>{
          const name = nameInput.value.trim();
          const phone = phoneInput.value.trim();
          const email = emailInput.value.trim();
          if(!name || !phone || !email){
            showAlert(alertInfo,"Name, phone, and email are required.");
            return;
          }
          currentDate = new Date();
          loadAvailabilityForDate(currentDate);
          showStep(4);
        });

        function formatDateTitle(d){
          return d.toLocaleDateString(undefined,{weekday:"short",month:"short",day:"numeric",year:"numeric"});
        }
        function formatTimeLabel(iso){
          const d = new Date(iso);
          return d.toLocaleTimeString(undefined,{hour:"numeric",minute:"2-digit"});
        }
        function groupForHour(h){
          if(h < 12) return "Morning";
          if(h < 17) return "Afternoon";
          return "Evening";
        }

        prevDayBtn.addEventListener("click",()=>{
          currentDate.setDate(currentDate.getDate()-1);
          loadAvailabilityForDate(currentDate);
        });
        nextDayBtn.addEventListener("click",()=>{
          currentDate.setDate(currentDate.getDate()+1);
          loadAvailabilityForDate(currentDate);
        });

        async function loadAvailabilityForDate(d){
          selectedSlot = null;
          renderSlotSelection();

          const isoDate = d.toISOString().slice(0,10);
          calendarTitle.textContent = formatDateTitle(d);

          try{
            const resp = await fetch("/api/availability",{
              method:"POST",
              headers:{"Content-Type":"application/json"},
              body:JSON.stringify({shop_id:shopId,date:isoDate})
            });
            if(!resp.ok) throw new Error("availability fail");
            const data = await resp.json();
            renderSlots(data.slots || []);
          }catch(e){
            console.error(e);
            showAlert(alertCalendar,"Unable to load availability. Try another date.");
          }
        }

        function renderSlots(slots){
          slotGroupsContainer.innerHTML = "";
          selectedSlot = null;

          if(!slots.length){
            slotEmpty.style.display = "block";
            return;
          }
          slotEmpty.style.display = "none";

          const groups = {Morning:[],Afternoon:[],Evening:[]};
          slots.forEach(s=>{
            const d = new Date(s.start);
            const h = d.getHours();
            const label = formatTimeLabel(s.start);
            const g = groupForHour(h);
            groups[g].push({label,raw:s});
          });

          Object.keys(groups).forEach(gname=>{
            const arr = groups[gname];
            if(!arr.length) return;
            const groupEl = document.createElement("div");
            groupEl.className = "sq-slot-group";
            const title = document.createElement("div");
            title.className = "sq-slot-group-title";
            title.textContent = gname;
            groupEl.appendChild(title);
            const wrap = document.createElement("div");
            wrap.className = "sq-slots";
            arr.forEach(item=>{
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "sq-slot";
              btn.textContent = item.label;
              btn.addEventListener("click",()=>{
                selectedSlot = item.raw;
                renderSlotSelection();
              });
              wrap.appendChild(btn);
            });
            groupEl.appendChild(wrap);
            slotGroupsContainer.appendChild(groupEl);
          });
        }

        function renderSlotSelection(){
          const all = slotGroupsContainer.querySelectorAll(".sq-slot");
          all.forEach(b=>b.classList.remove("selected"));
          if(!selectedSlot) return;
          all.forEach(b=>{
            if(b.textContent === formatTimeLabel(selectedSlot.start)){
              b.classList.add("selected");
            }
          });
        }

        bookBtn.addEventListener("click",async()=>{
          if(!selectedSlot){
            showAlert(alertCalendar,"Please select a time slot.");
            return;
          }
          if(!estimateId){
            showAlert(alertCalendar,"Please generate an estimate first.");
            showStep(1);
            return;
          }

          const vehicleText = makeModelInput.value.trim();
          const parts = vehicleText.split(" ");
          const make = parts[0] || null;
          const model = parts.slice(1).join(" ") || null;

          const payload = {
            shop_id:shopId,
            estimate_id:estimateId,
            customer_name:nameInput.value.trim(),
            phone:phoneInput.value.trim(),
            email:emailInput.value.trim(),
            vehicle_year:yearInput.value ? parseInt(yearInput.value,10) : null,
            vehicle_make:make,
            vehicle_model:model,
            slot_start:selectedSlot.start,
            slot_end:selectedSlot.end
          };

          setLoading(bookBtn,true,"Booking…");
          try{
            const resp = await fetch("/api/book",{
              method:"POST",
              headers:{"Content-Type":"application/json"},
              body:JSON.stringify(payload)
            });
            if(!resp.ok) throw new Error("booking failed");
            const data = await resp.json();
            const humanDate = formatDateTitle(new Date(selectedSlot.start));
            confirmText.textContent =
              "Your request has been sent. The shop will confirm your booking for " +
              humanDate + " and may reach out if they need anything else.";
            showStep(5);
          }catch(e){
            console.error(e);
            showAlert(alertCalendar,"We couldn’t complete your booking. Please try again.");
          }finally{
            setLoading(bookBtn,false);
          }
        });

        newRequestBtn.addEventListener("click",()=>{
          files = [];
          estimateId = null;
          selectedSlot = null;
          filePreview.innerHTML = "";
          nameInput.value = "";
          phoneInput.value = "";
          emailInput.value = "";
          yearInput.value = "";
          makeModelInput.value = "";
          showStep(1);
        });
      })();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
