import os
import json
import re
import base64
from datetime import datetime, timedelta, timezone, date, time
from typing import Optional, Dict, Any, List, Tuple
import uuid

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse, FileResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from dateutil import parser as date_parser

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Pydantic / settings
from pydantic import BaseModel, EmailStr
from pydantic_settings import BaseSettings

# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    SENDGRID_FROM_EMAIL: str = os.getenv("SENDGRID_FROM_EMAIL", "")
    DEFAULT_CALENDAR_ID: str = os.getenv("DEFAULT_CALENDAR_ID", "")
    GOOGLE_SERVICE_ACCOUNT_JSON: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://web-production-01391.up.railway.app")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

app = FastAPI()

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ============================================================
# GOOGLE CALENDAR SERVICE
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]
calendar_service = None

if settings.GOOGLE_SERVICE_ACCOUNT_JSON and settings.DEFAULT_CALENDAR_ID:
    try:
        creds_info = json.loads(settings.GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        print("[BOOT] Google Calendar ready.")
    except Exception as e:
        print("Error initializing Google Calendar:", e)
else:
    print("Google Calendar not configured (missing service account or calendar ID).")

# ============================================================
# TIMEZONE / DATE HELPERS
# ============================================================

DEFAULT_TZ = "America/Toronto"
LOCAL_TZ = timezone(timedelta(hours=-5))  # rough fallback

WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


# ============================================================
# SHOP CONFIG
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


def get_shop_by_token(token: str) -> ShopConfig:
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        raise HTTPException(status_code=404, detail="Shop not found for this token.")
    return shop


def get_shop_by_id(shop_id: str) -> ShopConfig:
    shop = SHOPS_BY_ID.get(shop_id)
    if not shop:
        raise HTTPException(status_code=404, detail="Shop not found.")
    return shop

# ============================================================
# Pydantic MODELS
# ============================================================

class DamageAnalysis(BaseModel):
    severity: str
    estimated_cost_min: float
    estimated_cost_max: float
    damaged_areas: List[str]
    damage_types: List[str]
    summary: str


class TimeSlot(BaseModel):
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return self.start.strftime("%-I:%M %p")


class AvailabilityRequest(BaseModel):
    shop_id: str
    date: date


class AvailabilityResponse(BaseModel):
    shop_id: str
    date: date
    slots: List[TimeSlot]


class BookingRequest(BaseModel):
    shop_id: str
    name: str
    email: EmailStr
    phone: str
    vehicle: str
    notes: Optional[str] = None
    slot_start: datetime
    slot_end: datetime
    estimate_summary: Optional[str] = None
    estimate_range: Optional[str] = None
    damage_severity: Optional[str] = None


class BookingResponse(BaseModel):
    success: bool
    message: str
    calendar_event_id: Optional[str] = None

# ============================================================
# UTILS: HTTPX / TWILIO IMAGE DOWNLOAD
# ============================================================

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as client_http:
        auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        resp = await client_http.get(url, auth=auth, timeout=30)
        resp.raise_for_status()
        return resp.content

# ============================================================
# AI DAMAGE ESTIMATOR (IMAGES → JSON)
# ============================================================

async def analyze_damage(image_bytes_list: List[bytes]) -> Optional[Dict[str, Any]]:
    system_prompt = (
        "You are a professional Ontario 2026 auto-body damage estimator.\n"
        "Return STRICT JSON ONLY.\n"
        "- Use DRIVER POV for left/right.\n"
        "- Consider bumper covers, reinforcements, lamps, quarter panels, hood, trunk,\n"
        "  crash sensors, radiator support, rocker, and potential structural damage.\n"
        "- Include blend panels when refinish is required on adjacent panels.\n"
        "- Use updated 2026 pricing bands (CAD):\n"
        "    minor: 400–2200\n"
        "    moderate: 2200–6500\n"
        "    severe: 6500–15000+\n"
        "- Calibrate the estimate range to what most quality independent collision\n"
        "  centres in Ontario would charge in 2026, not the cheapest possible price.\n"
        "- JSON keys: severity, estimated_cost_min, estimated_cost_max,\n"
        "  damaged_areas, damage_types, summary.\n"
    )

    content = [{"type": "text", "text": "Analyze the vehicle damage from these photos."}]
    for bytes_ in image_bytes_list:
        b64 = base64.b64encode(bytes_).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        return {
            "severity": data.get("severity", "moderate"),
            "estimated_cost_min": float(data.get("estimated_cost_min", 0)),
            "estimated_cost_max": float(data.get("estimated_cost_max", 0)),
            "damaged_areas": data.get("damaged_areas", []),
            "damage_types": data.get("damage_types", []),
            "summary": data.get("summary", ""),
        }
    except Exception as e:
        print("Error in analyze_damage:", e)
        return None

# ============================================================
# DATE / HOURS / AVAILABILITY HELPERS
# ============================================================

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
        except Exception:
            open_h, open_m = 9, 0
            close_h, close_m = 17, 0
    else:
        # Default fallback: Mon–Fri 9–5, Sat 10–2, Sun closed
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
        intervals: List[Tuple[datetime, datetime]] = []
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

    slots: List[TimeSlot] = []
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

# =============================
# BOOKING: CREATE CALENDAR EVENT
# =============================

def create_calendar_event(shop: ShopConfig, booking: BookingRequest) -> Optional[str]:
    if not calendar_service or not shop.calendar_id:
        print("Calendar service not configured or calendar_id missing.")
        return None

    event_body = {
        "summary": f"{shop.name} – AI Estimate Booking – {booking.name}",
        "description": (
            f"Customer: {booking.name}\n"
            f"Phone: {booking.phone}\n"
            f"Email: {booking.email}\n"
            f"Vehicle: {booking.vehicle}\n\n"
            f"Damage severity: {booking.damage_severity or 'n/a'}\n"
            f"Estimate range: {booking.estimate_range or 'n/a'}\n"
            f"Summary: {booking.estimate_summary or 'n/a'}\n\n"
            f"Notes from customer:\n{booking.notes or 'None'}\n"
        ),
        "start": {
            "dateTime": booking.slot_start.isoformat(),
            "timeZone": DEFAULT_TZ,
        },
        "end": {
            "dateTime": booking.slot_end.isoformat(),
            "timeZone": DEFAULT_TZ,
        },
    }

    try:
        event = calendar_service.events().insert(
            calendarId=shop.calendar_id,
            body=event_body,
            sendUpdates="all",
        ).execute()
        print("Created calendar event:", event.get("id"))
        return event.get("id")
    except Exception as e:
        print("Error creating calendar event:", e)
        return None

# ============================================================
# EMAIL NOTIFICATION (SendGrid)
# ============================================================

def send_booking_email_to_shop(shop: ShopConfig, booking: BookingRequest, event_id: Optional[str]):
    if not settings.SENDGRID_API_KEY or not settings.SENDGRID_FROM_EMAIL:
        print("SendGrid not configured; skipping booking email.")
        return

    subject = f"New SimpleQuotez booking – {booking.name}"
    event_link = (
        f"https://calendar.google.com/calendar/r/eventedit/{event_id}"
        if event_id
        else "Event ID not available"
    )

    html_content = f"""
    <h2>New booking for {shop.name}</h2>
    <p><strong>Name:</strong> {booking.name}</p>
    <p><strong>Phone:</strong> {booking.phone}</p>
    <p><strong>Email:</strong> {booking.email}</p>
    <p><strong>Vehicle:</strong> {booking.vehicle}</p>
    <p><strong>Requested time:</strong> {booking.slot_start.strftime('%Y-%m-%d %I:%M %p')} – {booking.slot_end.strftime('%I:%M %p')}</p>
    <p><strong>Damage severity:</strong> {booking.damage_severity or 'n/a'}</p>
    <p><strong>Estimate range:</strong> {booking.estimate_range or 'n/a'}</p>
    <p><strong>Summary:</strong><br>{(booking.estimate_summary or '').replace('\n', '<br>')}</p>
    <p><strong>Customer notes:</strong><br>{(booking.notes or 'None').replace('\n', '<br>')}</p>
    <p><strong>Calendar event:</strong> {event_link}</p>
    """

    message = Mail(
        from_email=settings.SENDGRID_FROM_EMAIL,
        to_emails=settings.SENDGRID_FROM_EMAIL,
        subject=subject,
        html_content=html_content,
    )

    try:
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        resp = sg.send(message)
        print("SendGrid email status:", resp.status_code)
    except Exception as e:
        print("Error sending SendGrid email:", e)

# ============================================================
# BOOKING & AVAILABILITY API (WEB)
# ============================================================

@app.post("/api/availability", response_model=AvailabilityResponse)
def api_availability(payload: AvailabilityRequest):
    """
    Safe wrapper around get_availability_for_shop so that any internal
    calendar or hours error still returns a valid (possibly empty) list
    instead of a 500 to the frontend.
    """
    try:
        slots = get_availability_for_shop(payload.shop_id, payload.date)
    except Exception as e:
        print("AVAILABILITY ERROR:", e)
        slots = []
    return AvailabilityResponse(
        shop_id=payload.shop_id,
        date=payload.date,
        slots=slots,
    )

@app.post("/api/book", response_model=BookingResponse)
def api_book(booking: BookingRequest):
    shop = get_shop_by_id(booking.shop_id)

    event_id = create_calendar_event(shop, booking)
    send_booking_email_to_shop(shop, booking, event_id)

    return BookingResponse(
        success=True,
        message="Appointment booked. You'll receive a confirmation shortly.",
        calendar_event_id=event_id,
    )

# ============================================================
# SMS FLOW (Twilio Webhook)
# ============================================================

WELCOME_MESSAGE = (
    "Welcome to SimpleQuotez – AI Damage Estimator.\n\n"
    "Send 1–3 clear photos of the damage, and I'll estimate the repair cost."
)

BOOKING_INSTRUCTIONS = (
    "If you’d like to book an in-person visit, reply with:\n"
    "BOOK + your full name, email, phone number, vehicle, and preferred date/time."
)

def normalize_datetime_phrase(dt: str) -> str:
    dt = dt.lower().strip()
    dt = dt.replace("tonight", "today 6pm")
    dt = dt.replace("this evening", "today 6pm")
    dt = dt.replace("tomorrow morning", "tomorrow 9am")
    dt = dt.replace("tomorrow afternoon", "tomorrow 1pm")
    dt = dt.replace("tomorrow evening", "tomorrow 6pm")
    dt = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", dt)
    return re.sub(r"\s+", " ", dt).strip()

def parse_booking_message(body: str) -> Dict[str, str]:
    """
    Very rough parser: expects something like:
    BOOK John Doe, john@email.com, 416-555-1234, 2018 Honda Civic, tomorrow 3pm
    """
    try:
        raw = body.strip()[4:].strip()  # strip "BOOK"
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        info = {
            "name": parts[0] if len(parts) > 0 else "",
            "email": parts[1] if len(parts) > 1 else "",
            "phone": parts[2] if len(parts) > 2 else "",
            "vehicle": parts[3] if len(parts) > 3 else "",
            "datetime_raw": parts[4] if len(parts) > 4 else "",
        }
        if info["datetime_raw"]:
            dt_str = normalize_datetime_phrase(info["datetime_raw"])
            try:
                parsed_dt = date_parser.parse(dt_str, fuzzy=True)
                info["datetime_parsed"] = parsed_dt
            except Exception:
                info["datetime_parsed"] = None
        return info
    except Exception:
        return {}

LAST_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From", "")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia") or 0)
    media_urls = [form.get(f"MediaUrl{i}") for i in range(num_media)]

    shop_token = request.query_params.get("token", "")
    shop = get_shop_by_token(shop_token)

    resp = MessagingResponse()
    msg = resp.message()

    if num_media > 0:
        image_bytes_list: List[bytes] = []
        for url in media_urls:
            if not url:
                continue
            try:
                image_bytes_list.append(await download_twilio_image(url))
            except Exception as e:
                print("Error downloading Twilio media:", e)

        if not image_bytes_list:
            msg.body("I couldn't download the images. Please try again.")
            return Response(content=str(resp), media_type="application/xml")

        analysis = await analyze_damage(image_bytes_list)
        if not analysis:
            msg.body(
                "Something went wrong analyzing the photos. Please try again, "
                "or contact the shop directly."
            )
            return Response(content=str(resp), media_type="application/xml")

        LAST_ANALYSIS_CACHE[from_number] = analysis

        severity = analysis["severity"]
        est_min = analysis["estimated_cost_min"]
        est_max = analysis["estimated_cost_max"]
        areas = ", ".join(analysis.get("damaged_areas", [])) or "N/A"
        summary = analysis.get("summary", "")

        text = (
            f"AI Damage Estimate for {shop.name}\n\n"
            f"Severity: {severity.capitalize()}\n"
            f"Estimated Range (Ontario 2026): "
            f"${est_min:,.0f} – ${est_max:,.0f}\n"
            f"Areas: {areas}\n\n"
            f"{summary}\n\n"
            f"This is a preliminary visual estimate only. Final pricing is "
            f"confirmed after an in-person inspection.\n\n"
            f"If you’d like to book an appointment, reply with:\n"
            f"BOOK + name, email, phone, vehicle, preferred date/time."
        )

        msg.body(text)
        return Response(content=str(resp), media_type="application/xml")

    if body.upper().startswith("BOOK"):
        info = parse_booking_message(body)
        if not info.get("name") or not info.get("email"):
            msg.body(
                "I couldn't understand your booking details.\n"
                "Please send: BOOK John Doe, john@email.com, 416-555-1234, "
                "2018 Honda Civic, tomorrow 3pm"
            )
            return Response(content=str(resp), media_type="application/xml")

        dt_parsed = info.get("datetime_parsed")
        if not dt_parsed:
            msg.body("I couldn't understand the date/time. Please try again.")
            return Response(content=str(resp), media_type="application/xml")

        slot_start = dt_parsed.replace(second=0, microsecond=0)
        slot_end = slot_start + timedelta(minutes=30)

        analysis = LAST_ANALYSIS_CACHE.get(from_number, {})
        est_range = ""
        if analysis:
            est_min = analysis.get("estimated_cost_min", 0)
            est_max = analysis.get("estimated_cost_max", 0)
            est_range = f"${est_min:,.0f} – ${est_max:,.0f}"

        booking = BookingRequest(
            shop_id=shop.id,
            name=info["name"],
            email=info["email"],
            phone=info["phone"],
            vehicle=info["vehicle"],
            notes=None,
            slot_start=slot_start,
            slot_end=slot_end,
            estimate_summary=analysis.get("summary") if analysis else None,
            estimate_range=est_range or None,
            damage_severity=analysis.get("severity") if analysis else None,
        )

        event_id = create_calendar_event(shop, booking)
        send_booking_email_to_shop(shop, booking, event_id)

        msg.body(
            f"Thanks {booking.name}, your visit request has been recorded for "
            f"{slot_start.strftime('%Y-%m-%d %I:%M %p')}. The shop will confirm shortly."
        )
        return Response(content=str(resp), media_type="application/xml")

    msg.body(WELCOME_MESSAGE)
    return Response(content=str(resp), media_type="application/xml")

# ============================================================
# STATIC: LOGO FILE
# ============================================================

@app.get("/logo.png")
async def get_logo():
    """
    Serves the SimpleQuotez logo. Place a file named 'logo.png'
    in the same directory as main.py.
    """
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if not os.path.exists(logo_path):
        # Fallback: 1x1 transparent PNG
        transparent_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )
        return Response(content=transparent_png, media_type="image/png")
    return FileResponse(logo_path)

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
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>SimpleQuotez – AI Damage Estimator</title>
      <style>
        :root{
          --sq-bg:#020617;
          --sq-card:#02091a;
          --sq-card-soft:#050c21;
          --sq-border:#1f2937;
          --sq-border-soft:rgba(148,163,184,.3);
          --sq-accent:#2563eb;
          --sq-accent-soft:#1d4ed8;
          --sq-accent-strong:#60a5fa;
          --sq-danger:#f97373;
          --sq-text-main:#e5e7eb;
          --sq-text-muted:#9ca3af;
          --sq-radius-lg:22px;
          --sq-radius-md:14px;
          --sq-radius-pill:999px;
          --sq-shadow-soft:0 18px 45px rgba(15,23,42,.65);
        }
        *{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Inter",sans-serif;}
        body{
          min-height:100vh;
          background:radial-gradient(circle at top,#0b1120,#000);
          color:var(--sq-text-main);
          display:flex;
          align-items:center;
          justify-content:center;
          padding:18px 10px 28px;
        }
        .sq-shell{
          width:100%;
          max-width:440px;
        }
        .sq-card{
          border-radius:30px;
          background:radial-gradient(circle at top left,#0b1120,#020617 45%,#000 110%);
          border:1px solid rgba(148,163,184,.25);
          box-shadow:var(--sq-shadow-soft);
          padding:18px 16px 20px;
          position:relative;
          overflow:hidden;
        }
        .sq-card::before{
          content:"";
          position:absolute;
          inset:-120px;
          background:
            radial-gradient(circle at 0 0,rgba(59,130,246,.16),transparent 55%),
            radial-gradient(circle at 100% 20%,rgba(147,51,234,.14),transparent 55%);
          opacity:.85;
          pointer-events:none;
        }
        .sq-card-inner{
          position:relative;
          z-index:1;
        }
        .sq-header{
          display:flex;
          align-items:center;
          justify-content:space-between;
          margin-bottom:14px;
        }
        .sq-header-left{
          display:flex;
          align-items:center;
          gap:10px;
        }
        .sq-logo-mark{
          width:32px;height:32px;border-radius:999px;
          display:flex;align-items:center;justify-content:center;
          background:radial-gradient(circle at 30% 0,rgba(59,130,246,.55),#020617);
          border:1px solid rgba(148,163,184,.5);
          box-shadow:0 0 20px rgba(37,99,235,.6);
          overflow:hidden;
        }
        .sq-header-text{
          display:flex;
          flex-direction:column;
          gap:2px;
        }
        .sq-header-title{
          font-size:15px;
          letter-spacing:.16em;
          text-transform:uppercase;
          color:#e5e7eb;
        }
        .sq-header-sub{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-step-dots{
          display:flex;
          align-items:center;
          gap:6px;
        }
        .sq-step-dot{
          width:7px;height:7px;border-radius:999px;
          background:rgba(148,163,184,.45);
        }
        .sq-step-dot.is-active{
          width:14px;
          background:linear-gradient(to right,#60a5fa,#4ade80);
        }
        .sq-step-dot.is-done{
          background:rgba(96,165,250,.9);
        }
        .sq-main-title{
          font-size:20px;
          font-weight:600;
          margin-bottom:6px;
        }
        .sq-main-sub{
          font-size:13px;
          line-height:1.5;
          color:var(--sq-text-muted);
          margin-bottom:16px;
        }
        .sq-stack{
          display:flex;
          flex-direction:column;
          gap:12px;
        }
        .sq-dropzone-row{
          display:flex;
          gap:10px;
        }
        .sq-dropzone-main{
          flex:1;
        }
        .sq-chip{
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:4px 9px;
          border-radius:999px;
          border:1px solid rgba(148,163,184,.45);
          background:rgba(15,23,42,.85);
          font-size:11px;
          color:var(--sq-text-muted);
          margin-bottom:8px;
        }
        .sq-chip-dot{
          width:6px;height:6px;border-radius:999px;
          background:radial-gradient(circle at 30% 0,#22c55e,#15803d);
        }
        .sq-upload-area{
          border-radius:var(--sq-radius-lg);
          border:1px solid rgba(148,163,184,.35);
          background:#020617;
          padding:18px 14px 16px;
          text-align:center;
          cursor:pointer;
          transition:border .15s,background .15s,transform .1s;
        }
        .sq-upload-area:hover{
          border-color:rgba(59,130,246,.9);
          background:#020617;
          transform:translateY(-1px);
        }
        .sq-upload-icon{
          width:40px;height:40px;border-radius:999px;
          display:flex;align-items:center;justify-content:center;
          background:radial-gradient(circle at 30% 0,rgba(59,130,246,.55),#020617);
          margin:0 auto 6px;
        }
        .sq-upload-icon span{
          font-size:22px;
        }
        .sq-upload-title{
          font-size:14px;
          font-weight:500;
          margin-bottom:2px;
        }
        .sq-upload-sub{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-bottom:6px;
        }
        .sq-upload-hint{
          font-size:11px;
          color:rgba(148,163,184,.85);
        }
        .sq-preview-pill{
          width:90px;
          min-height:90px;
          border-radius:22px;
          border:1px solid rgba(148,163,184,.35);
          background:radial-gradient(circle at top,#0b1120,#020617 70%);
          display:flex;
          align-items:center;
          justify-content:center;
          overflow:hidden;
          position:relative;
        }
        .sq-preview-pill img{
          width:100%;
          height:100%;
          object-fit:cover;
        }
        .sq-preview-pill-placeholder{
          font-size:11px;
          color:var(--sq-text-muted);
          text-align:center;
          padding:10px 6px;
        }
        .sq-row{
          display:flex;
          justify-content:space-between;
          align-items:center;
          margin-top:12px;
        }
        .sq-row-text{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-pill-badge{
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:4px 9px;
          border-radius:999px;
          border:1px solid rgba(96,165,250,.7);
          background:rgba(15,23,42,.8);
          font-size:11px;
          color:var(--sq-accent-strong);
        }
        .sq-pill-dot{
          width:7px;height:7px;border-radius:999px;
          background:radial-gradient(circle at 30% 0,#60a5fa,#1d4ed8);
        }
        .sq-footer{
          display:flex;
          justify-content:space-between;
          align-items:center;
          margin-top:16px;
        }
        .sq-footer-note{
          font-size:11px;
          color:var(--sq-text-muted);
          max-width:58%;
        }
        .sq-btn{
          border:none;
          border-radius:999px;
          padding:10px 18px;
          font-size:13px;
          font-weight:600;
          display:inline-flex;
          align-items:center;
          justify-content:center;
          gap:6px;
          cursor:pointer;
          transition:transform .08s,box-shadow .08s,background .12s;
        }
        .sq-btn-primary{
          background:linear-gradient(to right,#2563eb,#1d4ed8);
          color:white;
          box-shadow:0 12px 26px rgba(37,99,235,.55);
        }
        .sq-btn-primary:active{
          transform:translateY(1px);
          box-shadow:0 6px 14px rgba(37,99,235,.6);
        }
        .sq-btn-ghost{
          background:transparent;
          color:var(--sq-text-muted);
          border:1px solid rgba(148,163,184,.4);
        }
        .sq-btn-ghost:active{
          transform:translateY(1px);
        }
        .sq-subtle{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-top:10px;
        }

        /* Estimate step */
        .sq-estimate-chip{
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:4px 9px;
          border-radius:999px;
          border:1px solid rgba(96,165,250,.45);
          background:rgba(15,23,42,.9);
          font-size:11px;
        }
        .sq-severity-pill{
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:5px 10px;
          border-radius:999px;
          font-size:12px;
          border:1px solid rgba(234,179,8,.65);
          background:rgba(23,23,23,.9);
          color:#facc15;
          margin-bottom:8px;
        }
        .sq-summary-box{
          border-radius:18px;
          border:1px solid rgba(31,41,55,.85);
          background:radial-gradient(circle at top,#020617,#020617 60%,#020617);
          padding:10px 10px 11px;
          font-size:13px;
          color:rgba(229,231,235,.92);
          line-height:1.45;
        }
        .sq-summary-label{
          font-size:12px;
          font-weight:600;
          margin-bottom:4px;
        }

        /* Booking step */
        .sq-form-grid{
          display:flex;
          flex-direction:column;
          gap:10px;
          margin-top:8px;
        }
        .sq-field{
          display:flex;
          flex-direction:column;
          gap:4px;
        }
        .sq-label{
          font-size:12px;
          color:var(--sq-text-muted);
        }
        .sq-input{
          border-radius:999px;
          border:1px solid var(--sq-border-soft);
          background:rgba(15,23,42,.9);
          padding:8px 11px;
          font-size:13px;
          color:var(--sq-text-main);
          outline:none;
          transition:border .12s,box-shadow .12s,background .12s;
        }
        .sq-input:focus{
          border-color:rgba(59,130,246,.9);
          box-shadow:0 0 0 1px rgba(59,130,246,.6);
          background:#020617;
        }
        .sq-textarea{
          border-radius:16px;
          resize:vertical;
          min-height:56px;
        }

        /* Calendar step */
        .sq-calendar-shell{
          margin-top:8px;
          border-radius:18px;
          border:1px solid rgba(31,41,55,.8);
          background:radial-gradient(circle at top,#020617,#020617 65%,#020617);
          padding:10px 10px 11px;
        }
        .sq-calendar-header{
          display:flex;
          justify-content:space-between;
          align-items:center;
          margin-bottom:8px;
        }
        .sq-calendar-title{
          font-size:13px;
          font-weight:500;
        }
        .sq-calendar-nav{
          display:flex;
          align-items:center;
          gap:4px;
        }
        .sq-nav-btn{
          width:24px;height:24px;border-radius:999px;
          border:1px solid rgba(148,163,184,.45);
          background:rgba(15,23,42,.9);
          display:flex;
          align-items:center;
          justify-content:center;
          font-size:13px;
          cursor:pointer;
        }
        .sq-slot-groups{
          display:flex;
          flex-wrap:wrap;
          gap:6px;
          margin-top:8px;
        }
        .sq-slot-pill{
          padding:5px 10px;
          border-radius:999px;
          border:1px solid rgba(148,163,184,.55);
          font-size:12px;
          color:var(--sq-text-main);
          cursor:pointer;
          background:rgba(15,23,42,.95);
        }
        .sq-slot-pill.is-selected{
          border-color:rgba(59,130,246,.9);
          background:linear-gradient(to right,#2563eb,#1d4ed8);
        }
        .sq-slot-pill.is-disabled{
          opacity:.35;
          cursor:not-allowed;
        }
        .sq-calendar-note{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-top:6px;
        }

        /* Alerts */
        .sq-alert{
          border-radius:14px;
          padding:7px 9px;
          font-size:11px;
          margin-top:10px;
          display:none;
        }
        .sq-alert-error{
          border:1px solid rgba(248,113,113,.75);
          background:rgba(127,29,29,.85);
          color:#fee2e2;
        }
        .sq-alert-success{
          border:1px solid rgba(34,197,94,.8);
          background:rgba(22,163,74,.9);
          color:#dcfce7;
        }

        @media (min-width:480px){
          .sq-card{padding:20px 18px 22px;}
          .sq-main-title{font-size:21px;}
        }
      </style>
    </head>
    <body>
      <div class="sq-shell">
        <div class="sq-card">
          <div class="sq-card-inner">
            <header class="sq-header">
              <div class="sq-header-left">
                <div class="sq-logo-mark"><img src="/logo.png" alt="SimpleQuotez logo" style="width:26px;height:26px;border-radius:50%;object-fit:contain;"></div>
                <div class="sq-header-text">
                  <div class="sq-header-title">SIMPLEQUOTEZ</div>
                  <div class="sq-header-sub">AI Damage Estimator</div>
                </div>
              </div>
              <div class="sq-step-dots" id="stepDots">
                <div class="sq-step-dot is-active"></div>
                <div class="sq-step-dot"></div>
                <div class="sq-step-dot"></div>
                <div class="sq-step-dot"></div>
              </div>
            </header>

            <main id="stepContainer">
              <!-- Step content injected by JS -->
            </main>
          </div>
        </div>
      </div>

      <script>
        const shopId = new URLSearchParams(window.location.search).get("shop_id") || "miss";

        let currentStep = 0;
        let uploadedFiles = [];
        let aiEstimate = null;
        let selectedSlot = null;
        let selectedDate = new Date();
        let isSubmitting = false;

        const stepDots = document.getElementById("stepDots");
        const stepContainer = document.getElementById("stepContainer");

        function setStep(index){
          currentStep = index;
          Array.from(stepDots.children).forEach((dot,i)=>{
            dot.classList.remove("is-active","is-done");
            if(i < index) dot.classList.add("is-done");
            else if(i === index) dot.classList.add("is-active");
          });
          renderStep();
        }

        function formatCurrency(v){
          if(!v || isNaN(v)) return "$—";
          return new Intl.NumberFormat("en-CA",{
            style:"currency",
            currency:"CAD",
            maximumFractionDigits:0
          }).format(v);
        }

        function renderStep(){
          if(currentStep === 0){
            renderStepUpload();
          }else if(currentStep === 1){
            renderStepEstimate();
          }else if(currentStep === 2){
            renderStepForm();
          }else if(currentStep === 3){
            renderStepCalendar();
          }
        }

        function showAlert(el,msg,isError=true){
          el.textContent = msg;
          el.classList.remove("sq-alert-error","sq-alert-success");
          el.classList.add(isError ? "sq-alert-error" : "sq-alert-success");
          el.style.display = "block";
        }

        function clearAlert(el){
          el.textContent = "";
          el.style.display = "none";
        }

        function renderStepUpload(){
          stepContainer.innerHTML = `
            <section>
              <h1 class="sq-main-title">Get a fast repair estimate</h1>
              <p class="sq-main-sub">
                Upload 1–3 clear photos of the damage. Our AI will
                scan your vehicle and estimate the repair range.
              </p>

              <div class="sq-stack">
                <div class="sq-dropzone-row">
                  <div class="sq-dropzone-main">
                    <div class="sq-chip">
                      <div class="sq-chip-dot"></div>
                      <span>No account needed · Under a minute</span>
                    </div>
                    <label class="sq-upload-area" id="uploadZone">
                      <input id="fileInput" type="file" accept="image/*" multiple style="display:none" />
                      <div class="sq-upload-icon"><span>↑</span></div>
                      <div class="sq-upload-title">Drop photos here or tap to select</div>
                      <div class="sq-upload-sub">Front, angled, and close-up shots work best.</div>
                      <div class="sq-upload-hint">Up to 3 photos · JPG or PNG</div>
                    </label>
                  </div>
                  <div class="sq-preview-pill" id="previewPill">
                    <div class="sq-preview-pill-placeholder">
                      Your first photo will show here
                    </div>
                  </div>
                </div>

                <div class="sq-row">
                  <div class="sq-row-text">
                    Up to 3 photos · We’ll never share these outside the shop.
                  </div>
                  <div class="sq-pill-badge">
                    <div class="sq-pill-dot"></div>
                    <span>AI powered · 2026 pricing</span>
                  </div>
                </div>

                <div class="sq-footer">
                  <p class="sq-footer-note">
                    This estimate is visual only. Final pricing is confirmed
                    after a short in-person inspection at the shop.
                  </p>
                  <button id="btnAnalyze" class="sq-btn sq-btn-primary">
                    <span>Get AI estimate</span>
                  </button>
                </div>

                <p class="sq-subtle">
                  No downloads · No account · Estimate in under a minute.
                </p>

                <div class="sq-alert sq-alert-error" id="alertUpload"></div>
              </div>
            </section>
          `;

          const uploadZone = document.getElementById("uploadZone");
          const fileInput = document.getElementById("fileInput");
          const previewPill = document.getElementById("previewPill");
          const btnAnalyze = document.getElementById("btnAnalyze");
          const alertUpload = document.getElementById("alertUpload");

          function refreshPreview(){
            if(!uploadedFiles.length){
              previewPill.innerHTML = '<div class="sq-preview-pill-placeholder">Your first photo will show here</div>';
              return;
            }
            const file = uploadedFiles[0];
            const reader = new FileReader();
            reader.onload = e=>{
              previewPill.innerHTML = '<img src="'+e.target.result+'" alt="Damage photo" />';
            };
            reader.readAsDataURL(file);
          }

          uploadZone.addEventListener("click",()=>{
            fileInput.click();
          });

          uploadZone.addEventListener("dragover",e=>{
            e.preventDefault();
          });

          uploadZone.addEventListener("drop",e=>{
            e.preventDefault();
            const files = Array.from(e.dataTransfer.files).filter(f=>f.type.startsWith("image/")).slice(0,3);
            if(!files.length){
              showAlert(alertUpload,"Please drop image files only.");
              return;
            }
            clearAlert(alertUpload);
            uploadedFiles = files;
            refreshPreview();
          });

          fileInput.addEventListener("change",()=>{
            const files = Array.from(fileInput.files || []).filter(f=>f.type.startsWith("image/")).slice(0,3);
            if(!files.length){
              showAlert(alertUpload,"Please choose at least one image.");
              return;
            }
            clearAlert(alertUpload);
            uploadedFiles = files;
            refreshPreview();
          });

          btnAnalyze.addEventListener("click",async ()=>{
            clearAlert(alertUpload);
            if(!uploadedFiles.length){
              showAlert(alertUpload,"Please add at least one photo to analyze.");
              return;
            }

            btnAnalyze.disabled = true;
            btnAnalyze.innerHTML = "<span>Analyzing…</span>";

            const formData = new FormData();
            uploadedFiles.forEach((file,i)=>{
              formData.append("files",file);
            });
            formData.append("shop_id",shopId);

            try{
              const resp = await fetch("/web-analyze",{
                method:"POST",
                body:formData
              });
              if(!resp.ok) throw new Error("analysis failed");
              const data = await resp.json();
              aiEstimate = data;
              setStep(1);
            }catch(e){
              console.error(e);
              showAlert(alertUpload,"We couldn't analyze the photos. Please try again.");
            }finally{
              btnAnalyze.disabled = false;
              btnAnalyze.innerHTML = "<span>Get AI estimate</span>";
            }
          });

          refreshPreview();
        }

        function renderStepEstimate(){
          if(!aiEstimate){
            setStep(0);
            return;
          }
          const sev = (aiEstimate.severity || "").toLowerCase();
          let sevLabel = "Moderate";
          let sevColor = "#facc15";
          if(sev === "minor"){ sevLabel = "Minor"; sevColor = "#22c55e"; }
          if(sev === "severe"){ sevLabel = "Severe"; sevColor = "#fb7185"; }

          const rangeLabel = formatCurrency(aiEstimate.estimated_cost_min) + " – " +
                             formatCurrency(aiEstimate.estimated_cost_max);

          stepContainer.innerHTML = `
            <section>
              <h1 class="sq-main-title">Your AI estimate</h1>
              <p class="sq-main-sub">
                This is a preliminary visual estimate. Final pricing is confirmed
                after a short in-person inspection.
              </p>

              <div class="sq-stack">
                <div>
                  <div class="sq-severity-pill" style="border-color:${sevColor};color:${sevColor};">
                    <span>●</span>
                    <span>Severity: ${sevLabel}</span>
                  </div>
                  <div style="margin-top:4px;font-size:12px;">
                    Est. range: <strong>${rangeLabel}</strong>
                  </div>
                </div>

                <div style="font-size:12px;">
                  Areas: <span style="color:var(--sq-text-muted);">
                    ${(aiEstimate.damaged_areas || []).join(", ") || "N/A"}
                  </span>
                </div>

                <div class="sq-summary-box">
                  <div class="sq-summary-label">Summary</div>
                  <div>${(aiEstimate.summary || "").replace(/\\n/g,"<br>")}</div>
                </div>

                <div class="sq-footer">
                  <button class="sq-btn sq-btn-ghost" id="btnBackUpload">
                    Back
                  </button>
                  <button class="sq-btn sq-btn-primary" id="btnContinueBooking">
                    Continue to booking
                  </button>
                </div>

                <p class="sq-subtle">
                  You'll only confirm the visit after the shop reviews your photos and estimate.
                </p>
              </div>
            </section>
          `;

          document.getElementById("btnBackUpload").addEventListener("click",()=>{
            setStep(0);
          });
          document.getElementById("btnContinueBooking").addEventListener("click",()=>{
            setStep(2);
          });
        }

        function renderStepForm(){
          stepContainer.innerHTML = `
            <section>
              <h1 class="sq-main-title">Tell us a few details</h1>
              <p class="sq-main-sub">
                We'll share your estimate and photos with the shop so they can prepare
                for your visit.
              </p>

              <div class="sq-form-grid">
                <div class="sq-field">
                  <label class="sq-label">Full name</label>
                  <input id="nameInput" class="sq-input" placeholder="Jane Driver" />
                </div>
                <div class="sq-field">
                  <label class="sq-label">Email</label>
                  <input id="emailInput" class="sq-input" placeholder="you@example.com" />
                </div>
                <div class="sq-field">
                  <label class="sq-label">Mobile number</label>
                  <input id="phoneInput" class="sq-input" placeholder="(555) 123-4567" />
                </div>
                <div class="sq-field">
                  <label class="sq-label">Vehicle</label>
                  <input id="vehicleInput" class="sq-input" placeholder="2019 Toyota Corolla" />
                </div>
                <div class="sq-field">
                  <label class="sq-label">Anything the shop should know?</label>
                  <textarea id="notesInput" class="sq-input sq-textarea" placeholder="Rental needs, prior repairs, insurance claim #, etc."></textarea>
                </div>
              </div>

              <div class="sq-footer" style="margin-top:16px;">
                <button class="sq-btn sq-btn-ghost" id="btnBackEstimate">Back</button>
                <button class="sq-btn sq-btn-primary" id="btnContinueCalendar">
                  Select a time
                </button>
              </div>

              <div class="sq-alert sq-alert-error" id="alertForm"></div>
            </section>
          `;

          const nameInput = document.getElementById("nameInput");
          const emailInput = document.getElementById("emailInput");
          const phoneInput = document.getElementById("phoneInput");
          const vehicleInput = document.getElementById("vehicleInput");
          const notesInput = document.getElementById("notesInput");
          const alertForm = document.getElementById("alertForm");

          document.getElementById("btnBackEstimate").addEventListener("click",()=>{
            setStep(1);
          });

          document.getElementById("btnContinueCalendar").addEventListener("click",()=>{
            clearAlert(alertForm);
            const name = nameInput.value.trim();
            const email = emailInput.value.trim();
            const phone = phoneInput.value.trim();
            const vehicle = vehicleInput.value.trim();

            if(!name || !email || !phone || !vehicle){
              showAlert(alertForm,"Please fill out name, email, phone, and vehicle.");
              return;
            }

            window.__sqBookingForm = {
              name,
              email,
              phone,
              vehicle,
              notes: notesInput.value.trim()
            };

            setStep(3);
          });
        }

        function renderStepCalendar(){
          stepContainer.innerHTML = `
            <section>
              <h1 class="sq-main-title">Select a time</h1>
              <p class="sq-main-sub">
                Only open slots for this shop are shown. Times are local to the shop.
              </p>

              <div class="sq-calendar-shell">
                <div class="sq-calendar-header">
                  <div class="sq-calendar-title" id="calendarTitle"></div>
                  <div class="sq-calendar-nav">
                    <button class="sq-nav-btn" id="btnPrevDay"><</button>
                    <button class="sq-nav-btn" id="btnNextDay">></button>
                  </div>
                </div>
                <div class="sq-slot-groups" id="slotGroups"></div>
                <div class="sq-calendar-note" id="slotEmpty" style="display:none;">
                  No open slots for this day. Try another date.
                </div>
              </div>

              <div class="sq-footer" style="margin-top:14px;">
                <button class="sq-btn sq-btn-ghost" id="btnBackForm">Back</button>
                <button class="sq-btn sq-btn-primary" id="btnConfirmBooking">
                  Confirm appointment
                </button>
              </div>

              <div class="sq-alert sq-alert-error" id="alertCalendar"></div>
              <div class="sq-alert sq-alert-success" id="alertSuccess"></div>
            </section>
          `;

          const calendarTitle = document.getElementById("calendarTitle");
          const slotGroupsContainer = document.getElementById("slotGroups");
          const slotEmpty = document.getElementById("slotEmpty");
          const btnPrevDay = document.getElementById("btnPrevDay");
          const btnNextDay = document.getElementById("btnNextDay");
          const btnBackForm = document.getElementById("btnBackForm");
          const btnConfirmBooking = document.getElementById("btnConfirmBooking");
          const alertCalendar = document.getElementById("alertCalendar");
          const alertSuccess = document.getElementById("alertSuccess");

          function formatDateTitle(d){
            return d.toLocaleDateString("en-CA",{
              weekday:"short",
              month:"short",
              day:"numeric",
              year:"numeric"
            });
          }

          function renderSlotSelection(){
            Array.from(slotGroupsContainer.querySelectorAll(".sq-slot-pill")).forEach(el=>{
              const startIso = el.getAttribute("data-start");
              if(selectedSlot && selectedSlot.startIso === startIso){
                el.classList.add("is-selected");
              }else{
                el.classList.remove("is-selected");
              }
            });
          }

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

            const now = new Date();

            slots.forEach(s=>{
              const start = new Date(s.start);
              const end = new Date(s.end);
              const label = start.toLocaleTimeString("en-CA",{
                hour:"numeric",
                minute:"2-digit"
              });

              const btn = document.createElement("button");
              btn.className = "sq-slot-pill";
              btn.textContent = label;
              btn.setAttribute("data-start",s.start);
              btn.setAttribute("data-end",s.end);

              if(start < now){
                btn.classList.add("is-disabled");
              }else{
                btn.addEventListener("click",()=>{
                  selectedSlot = {
                    start,
                    end,
                    startIso:s.start,
                    endIso:s.end,
                    label
                  };
                  renderSlotSelection();
                });
              }

              slotGroupsContainer.appendChild(btn);
            });

            renderSlotSelection();
          }

          btnPrevDay.addEventListener("click",()=>{
            selectedDate.setDate(selectedDate.getDate()-1);
            loadAvailabilityForDate(selectedDate);
          });

          btnNextDay.addEventListener("click",()=>{
            selectedDate.setDate(selectedDate.getDate()+1);
            loadAvailabilityForDate(selectedDate);
          });

          btnBackForm.addEventListener("click",()=>{
            setStep(2);
          });

          btnConfirmBooking.addEventListener("click",async ()=>{
            clearAlert(alertCalendar);
            clearAlert(alertSuccess);

            if(!selectedSlot){
              showAlert(alertCalendar,"Please choose a time slot.");
              return;
            }
            if(!window.__sqBookingForm){
              showAlert(alertCalendar,"Form data missing. Go back one step.");
              return;
            }

            const payload = {
              shop_id:shopId,
              name:window.__sqBookingForm.name,
              email:window.__sqBookingForm.email,
              phone:window.__sqBookingForm.phone,
              vehicle:window.__sqBookingForm.vehicle,
              notes:window.__sqBookingForm.notes || "",
              slot_start:selectedSlot.startIso,
              slot_end:selectedSlot.endIso,
              estimate_summary:aiEstimate ? aiEstimate.summary : "",
              estimate_range:aiEstimate ? (
                formatCurrency(aiEstimate.estimated_cost_min) + " – " +
                formatCurrency(aiEstimate.estimated_cost_max)
              ) : "",
              damage_severity:aiEstimate ? aiEstimate.severity : ""
            };

            if(isSubmitting) return;
            isSubmitting = true;
            btnConfirmBooking.disabled = true;
            btnConfirmBooking.textContent = "Booking…";

            try{
              const resp = await fetch("/api/book",{
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body:JSON.stringify(payload)
              });
              if(!resp.ok) throw new Error("booking failed");
              const data = await resp.json();
              if(data.success){
                showAlert(alertSuccess,"Appointment booked. Watch for your confirmation email.",false);
              }else{
                showAlert(alertCalendar,data.message || "We couldn't book this slot. Please try again.");
              }
            }catch(e){
              console.error(e);
              showAlert(alertCalendar,"Something went wrong. Please try again.");
            }finally{
              isSubmitting = false;
              btnConfirmBooking.disabled = false;
              btnConfirmBooking.textContent = "Confirm appointment";
            }
          });

          loadAvailabilityForDate(selectedDate);
        }

        // Initialize
        setStep(0);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ============================================================
# WEB ANALYZE ENDPOINT (USED BY /quote)
# ============================================================

@app.post("/web-analyze")
async def web_analyze(
    shop_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    try:
        shop = get_shop_by_id(shop_id)
    except HTTPException:
        shop = None

    image_bytes_list: List[bytes] = []
    for f in files[:3]:
        try:
            image_bytes_list.append(await f.read())
        except Exception as e:
            print("Error reading upload file:", e)

    if not image_bytes_list:
        return {"error": "No valid images uploaded."}

    analysis = await analyze_damage(image_bytes_list)
    if not analysis:
        return {"error": "AI analysis failed."}

    return analysis

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "SimpleQuotez AI Estimator",
        "version": "simplequotez_full_v1_webapp",
        "timezone": DEFAULT_TZ,
    }

