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

settings = Settings()

DEFAULT_TZ = "America/Toronto"
TZ = timezone(timedelta(hours=-5))  # simple fixed offset for demo

# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: Dict[str, Any]
    hours: Dict[str, Any]

# Example single shop
SHOPS: List[ShopConfig] = [
    ShopConfig(
        id="miss",
        name="Mississauga Collision Centre",
        webhook_token="shop_miss_123",
        calendar_id="shiran.bookings@gmail.com",
        pricing={
            "labor_rates": {"body": 95, "paint": 105},
            "materials_rate": 38,
            "base_floor": {
                "minor_min": 350,
                "minor_max": 650,
                "moderate_min": 900,
                "moderate_max": 1600,
                "severe_min": 2000,
                "severe_max": 5000,
            },
        },
        hours={
            "monday": {"open": "09:00", "close": "17:00"},
            "tuesday": {"open": "09:00", "close": "17:00"},
            "wednesday": {"open": "09:00", "close": "17:00"},
            "thursday": {"open": "09:00", "close": "17:00"},
            "friday": {"open": "09:00", "close": "17:00"},
            "saturday": {"open": None, "close": None},
            "sunday": {"open": None, "close": None},
        },
    )
]

def get_shop_by_id(shop_id: str) -> ShopConfig:
    for s in SHOPS:
        if s.id == shop_id:
            return s
    raise HTTPException(status_code=404, detail="Shop not found")

# ============================================================
# APP + CLIENTS
# ============================================================

app = FastAPI()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ============================================================
# MODELS FOR WEB BOOKING
# ============================================================

class AvailabilitySlot(BaseModel):
    start: datetime
    end: datetime

class AvailabilityRequest(BaseModel):
    shop_id: str
    date: date

class AvailabilityResponse(BaseModel):
    shop_id: str
    date: date
    slots: List[AvailabilitySlot]

class BookingRequest(BaseModel):
    shop_id: str
    estimate_id: str
    customer_name: str
    phone: str
    email: EmailStr
    vehicle: str
    slot_start: datetime
    slot_end: datetime
    notes: Optional[str] = None

class BookingResponse(BaseModel):
    success: bool
    message: str
    calendar_event_id: Optional[str] = None

# ============================================================
# AI PROMPT + COST LOGIC (ONTARIO 2026)
# ============================================================

def build_ai_prompt() -> str:
    return (
        "You are a professional Ontario 2026 auto-body damage estimator.\n"
        "You estimate repair costs for collision damage from PHOTOS only.\n"
        "You work for high-quality, reputable collision centres in Ontario.\n"
        "Your job:\n"
        "1) Identify damaged areas (e.g., front bumper, rear bumper, trunk lid,\n"
        "   hood, fenders, doors, headlights, taillights, quarter panels).\n"
        "2) Identify damage types (dents, cracks, scrapes, panel deformation,\n"
        "   misalignment, paint damage, structural damage, etc.).\n"
        "3) Classify overall severity: minor, moderate, or severe.\n"
        "4) Estimate a realistic Ontario 2026 repair cost range in CAD.\n"
        "\n"
        "Important:\n"
        "- Use data consistent with what reputable collision centres in Ontario\n"
        "  would charge in 2026, not the cheapest possible price.\n"
        "- Minor: cosmetic & light panel work (typical range 350–1,000+ CAD).\n"
        "- Moderate: combination of panel work, parts replacement, paint\n"
        "  blending, possible sensor/calibration (900–3,000+ CAD).\n"
        "- Severe: multiple panels, structural areas, safety-critical components,\n"
        "  heavy labor & parts (2,000–10,000+ CAD depending on vehicle).\n"
        "- Always assume late-model everyday vehicles (not supercars) unless the\n"
        "  photo clearly shows a luxury/exotic vehicle.\n"
        "- If photos are unclear or missing angles, be conservative and say that\n"
        "  the range could change after an in-person inspection.\n"
        "\n"
        "Output JSON ONLY with keys:\n"
        "{\n"
        '  "severity": "minor|moderate|severe",\n'
        '  "areas": ["front bumper", "left fender", ...],\n'
        '  "damage_types": ["dent", "crack", ...],\n'
        '  "estimated_min": 1200,\n'
        '  "estimated_max": 3200,\n'
        '  "labor_hours": 10.5,\n'
        '  "notes": "short bullet-style reasoning",\n'
        '  "summary": "1–3 sentence summary for the customer."\n'
        "}\n"
    )

def clamp_cost_range(
    base_min: float, base_max: float, shop: ShopConfig, severity: str
) -> Tuple[float, float]:
    floor = shop.pricing.get("base_floor", {})
    if severity == "minor":
        mn = floor.get("minor_min", 350)
        mx = floor.get("minor_max", 650)
    elif severity == "moderate":
        mn = floor.get("moderate_min", 900)
        mx = floor.get("moderate_max", 1600)
    else:
        mn = floor.get("severe_min", 2000)
        mx = floor.get("severe_max", 5000)

    mn = min(max(base_min, mn), mx)
    mx = max(min(base_max, mx * 1.5), mn + 150)
    return round(mn, -1), round(mx, -1)

# ============================================================
# GOOGLE CALENDAR – MOCKED / SIMPLE VERSION
# ============================================================

def get_shop_hours_for_date(shop: ShopConfig, d: date) -> Optional[Tuple[time, time]]:
    weekday = d.weekday()  # 0=Mon
    key = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ][weekday]
    info = shop.hours.get(key)
    if not info or not info.get("open") or not info.get("close"):
        return None
    open_t = datetime.strptime(info["open"], "%H:%M").time()
    close_t = datetime.strptime(info["close"], "%H:%M").time()
    return open_t, close_t

def generate_daily_slots(
    shop: ShopConfig, d: date, duration_minutes: int = 30
) -> List[AvailabilitySlot]:
    hours = get_shop_hours_for_date(shop, d)
    if not hours:
        return []
    open_t, close_t = hours
    start_dt = datetime.combine(d, open_t, tzinfo=TZ)
    end_dt = datetime.combine(d, close_t, tzinfo=TZ)

    slots: List[AvailabilitySlot] = []
    cur = start_dt
    while cur + timedelta(minutes=duration_minutes) <= end_dt:
        slot_end = cur + timedelta(minutes=duration_minutes)
        # In a real system you’d also check existing calendar events here.
        slots.append(AvailabilitySlot(start=cur, end=slot_end))
        cur = slot_end

    # For same-day, remove past slots
    now = datetime.now(TZ)
    slots = [s for s in slots if s.start > now]
    return slots

def get_availability_for_shop(shop_id: str, d: date) -> List[AvailabilitySlot]:
    shop = get_shop_by_id(shop_id)
    return generate_daily_slots(shop, d)

def create_calendar_event(shop: ShopConfig, booking: BookingRequest) -> str:
    # Stub – in production you’d call Google Calendar API here
    fake_event_id = f"fake-{uuid.uuid4()}"
    print(
        f"[CALENDAR] Would create event {fake_event_id} for shop={shop.id}, "
        f"{booking.slot_start} – {booking.slot_end}"
    )
    return fake_event_id

# ============================================================
# SENDGRID EMAIL
# ============================================================

def send_booking_email_to_shop(shop: ShopConfig, booking: BookingRequest, event_id: str):
    if not settings.SENDGRID_API_KEY or not settings.SENDGRID_FROM_EMAIL:
        print("[EMAIL] Missing SendGrid config; skipping email.")
        return

    subject = f"[SimpleQuotez] New booking for {shop.name}"
    start_local = booking.slot_start.astimezone(TZ).strftime("%A, %b %d at %I:%M %p")
    html_content = f"""
    <h2>New SimpleQuotez Booking - {shop.name}</h2>
    <p><strong>Customer:</strong> {booking.customer_name}</p>
    <p><strong>Phone:</strong> {booking.phone}</p>
    <p><strong>Email:</strong> {booking.email}</p>
    <p><strong>Vehicle:</strong> {booking.vehicle}</p>
    <p><strong>Preferred time:</strong> {start_local} (local shop time)</p>
    <p><strong>Notes:</strong> {booking.notes or "-"} </p>
    <p><strong>Estimate ID:</strong> {booking.estimate_id}</p>
    <p><strong>Calendar ID:</strong> {shop.calendar_id}</p>
    <p><strong>Created Event ID (demo):</strong> {event_id}</p>
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
# SMS FLOW (Twilio Webhook) – still supported
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
    return dt.strip()

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN))
        r.raise_for_status()
        return r.content

async def run_ai_estimate_from_bytes_list(
    images: List[bytes], shop: ShopConfig
) -> Dict[str, Any]:
    if not images:
        raise ValueError("No images to analyze.")
    image_inputs = []
    for b in images[:3]:
        image_inputs.append(
            {
                "type": "input_image",
                "image": {
                    "data": base64.b64encode(b).decode("utf-8"),
                    "media_type": "image/jpeg",
                },
            }
        )

    system_prompt = build_ai_prompt()

    msg = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are looking at 1–3 photos of a damaged vehicle. "
                        "Please analyse the damage following the instructions."
                    ),
                },
                *image_inputs,
            ],
        },
    ]

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=msg,
        response_format={"type": "json_object"},
        max_output_tokens=600,
    )
    raw = resp.output[0].content[0].text
    try:
        data = json.loads(raw)
    except Exception:
        raise ValueError(f"AI JSON parse error: {raw}")

    severity = data.get("severity", "moderate").lower()
    est_min = float(data.get("estimated_min", 900))
    est_max = float(data.get("estimated_max", 1600))
    est_min, est_max = clamp_cost_range(est_min, est_max, shop, severity)

    data["estimated_min"] = est_min
    data["estimated_max"] = est_max
    return data

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From", "")
    num_media = int(form.get("NumMedia", "0"))

    resp = MessagingResponse()

    body = (form.get("Body", "") or "").strip()
    lower = body.lower()

    if num_media == 0 and not lower.startswith("book"):
        msg = resp.message(
            WELCOME_MESSAGE
            + "\n\n"
            "Reply with up to 3 photos of the damage to get a visual estimate."
        )
        return Response(content=str(resp), media_type="application/xml")

    if lower.startswith("book"):
        msg = resp.message(
            "Thanks! Your booking request has been received. A team member will confirm the exact time shortly."
        )
        return Response(content=str(resp), media_type="application/xml")

    images: List[bytes] = []
    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        if not media_url:
            continue
        try:
            img_bytes = await download_twilio_image(media_url)
            images.append(img_bytes)
        except Exception as e:
            print("Error downloading Twilio image:", e)

    if not images:
        resp.message(
            "I couldn't find any valid photos in your message. "
            "Please send 1–3 clear photos of the damage."
        )
        return Response(content=str(resp), media_type="application/xml")

    shop = SHOPS[0]

    try:
        analysis = await run_ai_estimate_from_bytes_list(images, shop)
    except Exception as e:
        print("AI error:", e)
        resp.message(
            "Sorry, there was an issue analysing your photos. Please try again later."
        )
        return Response(content=str(resp), media_type="application/xml")

    severity = analysis.get("severity", "moderate").capitalize()
    est_min = analysis.get("estimated_min")
    est_max = analysis.get("estimated_max")
    areas = analysis.get("areas", [])
    damage_types = analysis.get("damage_types", [])
    summary = analysis.get("summary", "")

    price_line = f"Estimated Range (Ontario 2026): ${est_min:,.0f} – ${est_max:,.0f}"
    areas_line = ", ".join(areas) if areas else "Not clearly visible"
    types_line = ", ".join(damage_types) if damage_types else "Not clearly visible"

    text = (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {severity}\n"
        f"{price_line}\n"
        f"Areas: {areas_line}\n"
        f"Damage Types: {types_line}\n\n"
        f"{summary}\n\n"
        "This is a visual, preliminary estimate only.\n\n"
        + BOOKING_INSTRUCTIONS
    )
    resp.message(text)
    return Response(content=str(resp), media_type="application/xml")

# ============================================================
# STATIC LOGO ENDPOINT
# ============================================================

@app.get("/logo.png")
async def get_logo():
    """
    Serves the SimpleQuotez logo. Place a file named 'logo.png'
    in the same folder as main.py (which you already did).
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
          --sq-card-soft:#020817;
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
        .sq-card-glow{
          position:absolute;
          inset:-40%;
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
          font-size:13px;
          letter-spacing:.16em;
          text-transform:uppercase;
          color:var(--sq-text-main);
        }
        .sq-header-sub{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-step-dots{
          display:flex;
          gap:4px;
        }
        .sq-step-dot{
          width:7px;
          height:7px;
          border-radius:999px;
          background:rgba(75,85,99,.7);
          transition:all .2s ease;
        }
        .sq-step-dot.is-active{
          width:16px;
          background:linear-gradient(90deg,#60a5fa,#a855f7);
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
          border:1px dashed rgba(148,163,184,.6);
          display:flex;align-items:center;justify-content:center;
          margin:0 auto 10px;
          font-size:18px;
          color:var(--sq-accent-strong);
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
          padding:10px;
          text-align:center;
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
          color:rgba(209,213,219,.95);
          margin-bottom:8px;
        }
        .sq-chip-dot{
          width:6px;height:6px;border-radius:999px;
          background:radial-gradient(circle,#4ade80,#22c55e);
          box-shadow:0 0 10px rgba(34,197,94,.7);
        }
        .sq-btn-row{
          display:flex;
          gap:8px;
        }
        .sq-btn{
          flex:1;
          border-radius:var(--sq-radius-pill);
          border:none;
          padding:9px 14px;
          font-size:13px;
          font-weight:500;
          cursor:pointer;
          display:flex;
          align-items:center;
          justify-content:center;
          gap:6px;
          transition:background .15s,transform .1s,box-shadow .15s;
        }
        .sq-btn-primary{
          background:linear-gradient(90deg,#2563eb,#4f46e5);
          color:white;
          box-shadow:0 10px 25px rgba(37,99,235,.7);
        }
        .sq-btn-primary:hover{
          background:linear-gradient(90deg,#1d4ed8,#4338ca);
          transform:translateY(-1px);
        }
        .sq-btn-ghost{
          background:rgba(15,23,42,.9);
          color:var(--sq-text-main);
          border:1px solid rgba(31,41,55,.85);
        }
        .sq-btn-ghost:hover{
          background:rgba(15,23,42,1);
        }
        .sq-footer{
          margin-top:14px;
          display:flex;
          justify-content:space-between;
          align-items:center;
          gap:10px;
        }
        .sq-footer-text{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-alert{
          margin-top:10px;
          font-size:11px;
          border-radius:12px;
          padding:7px 9px;
          display:none;
        }
        .sq-alert-error{
          background:rgba(248,113,113,.08);
          border:1px solid rgba(248,113,113,.5);
          color:#fecaca;
        }
        .sq-alert-success{
          background:rgba(34,197,94,.08);
          border:1px solid rgba(34,197,94,.5);
          color:#bbf7d0;
        }
        .sq-label{
          font-size:12px;
          color:var(--sq-text-muted);
          margin-bottom:3px;
        }
        .sq-input{
          width:100%;
          border-radius:var(--sq-radius-md);
          border:1px solid rgba(31,41,55,.85);
          background:rgba(15,23,42,.96);
          color:var(--sq-text-main);
          padding:8px 9px;
          font-size:13px;
        }
        .sq-input:focus{
          outline:none;
          border-color:rgba(59,130,246,.9);
          box-shadow:0 0 0 1px rgba(59,130,246,.3);
        }
        .sq-input-row{
          display:flex;
          gap:8px;
        }
        .sq-input-row > div{
          flex:1;
        }
        .sq-field{
          margin-bottom:8px;
        }
        .sq-textarea{
          min-height:48px;
          resize:vertical;
        }
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
          display:flex;align-items:center;justify-content:center;
          color:#9ca3af;
          cursor:pointer;
          font-size:14px;
        }
        .sq-nav-btn:hover{
          border-color:rgba(209,213,219,.85);
          color:#e5e7eb;
        }
        .sq-date-line{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-bottom:6px;
        }
        .sq-slot-groups{
          display:flex;
          flex-wrap:wrap;
          gap:6px;
        }
        .sq-slot-pill{
          border-radius:999px;
          border:1px solid rgba(55,65,81,.9);
          padding:5px 9px;
          font-size:11px;
          color:#e5e7eb;
          background:rgba(15,23,42,.95);
          cursor:pointer;
          transition:all .15s;
        }
        .sq-slot-pill:hover{
          border-color:rgba(59,130,246,.9);
        }
        .sq-slot-pill.is-selected{
          background:linear-gradient(90deg,#2563eb,#4f46e5);
          border-color:transparent;
          box-shadow:0 12px 22px rgba(37,99,235,.6);
        }
        .sq-slot-empty{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-top:4px;
        }
        .sq-summary-box{
          margin-top:10px;
          border-radius:16px;
          border:1px solid rgba(31,41,55,.85);
          background:radial-gradient(circle at top,#020617,#020617 65%,#020617);
          padding:10px 11px;
          font-size:12px;
        }
        .sq-summary-label{
          font-size:11px;
          color:var(--sq-text-muted);
          margin-bottom:4px;
        }
        .sq-summary-price{
          font-size:14px;
          font-weight:600;
          margin-bottom:3px;
        }
        .sq-summary-detail{
          font-size:12px;
          color:var(--sq-text-muted);
          margin-bottom:2px;
        }
        .sq-summary-badges{
          margin-top:6px;
          display:flex;
          flex-wrap:wrap;
          gap:6px;
        }
        .sq-badge{
          border-radius:999px;
          border:1px solid rgba(55,65,81,.9);
          padding:3px 7px;
          font-size:10px;
          color:#e5e7eb;
          background:rgba(15,23,42,.95);
        }
        .sq-subtle{
          margin-top:10px;
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-footer-actions{
          display:flex;
          justify-content:flex-end;
          gap:8px;
          margin-top:10px;
        }
        .sq-kicker{
          font-size:10px;
          text-transform:uppercase;
          letter-spacing:.18em;
          color:rgba(148,163,184,.95);
          margin-bottom:4px;
        }
        .sq-tagline{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-row{
          display:flex;
          justify-content:space-between;
          align-items:flex-start;
          gap:10px;
        }
        .sq-row-text{
          font-size:11px;
          color:var(--sq-text-muted);
        }
        .sq-pill-badge{
          border-radius:999px;
          border:1px solid rgba(55,65,81,.9);
          padding:4px 9px;
          font-size:10px;
          display:flex;
          align-items:center;
          gap:6px;
          color:rgba(209,213,219,.95);
        }
        .sq-pill-dot{
          width:7px;height:7px;border-radius:999px;
          background:radial-gradient(circle,#38bdf8,#0369a1);
          box-shadow:0 0 10px rgba(56,189,248,.8);
        }
        .sq-divider{
          margin:10px 0;
          height:1px;
          background:linear-gradient(90deg,transparent,rgba(75,85,99,.9),transparent);
        }
        @media (max-width:360px){
          .sq-shell{max-width:100%;}
          .sq-card{padding:16px 12px 18px;}
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
          <div class="sq-card-glow"></div>
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
        let selectedDate = null;
        let selectedSlot = null;
        let estimateId = null;

        const stepDots = document.getElementById("stepDots");
        const stepContainer = document.getElementById("stepContainer");

        function setStep(step){
          currentStep = step;
          Array.from(stepDots.children).forEach((dot,idx)=>{
            if(idx<=step){
              dot.classList.add("is-active");
            }else{
              dot.classList.remove("is-active");
            }
          });

          if(step===0){
            renderStepUpload();
          }else if(step===1){
            renderStepEstimate();
          }else if(step===2){
            renderStepForm();
          }else if(step===3){
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

        function formatCurrency(num){
          return new Intl.NumberFormat("en-CA",{style:"currency",currency:"CAD",maximumFractionDigits:0}).format(num);
        }

        function formatDateLabel(d){
          return d.toLocaleDateString("en-CA",{
            weekday:"short",
            month:"short",
            day:"numeric",
            year:"numeric"
          });
        }

        async function callWebAnalyze(){
          if(!uploadedFiles.length){
            return {error:"No files"};
          }
          const formData = new FormData();
          formData.append("shop_id",shopId);
          uploadedFiles.forEach((f,i)=>{
            formData.append("files",f);
          });

          const resp = await fetch("/web-analyze",{
            method:"POST",
            body:formData
          });
          if(!resp.ok){
            return {error:"Server error"};
          }
          return await resp.json();
        }

        function renderStepUpload(){
          stepContainer.innerHTML = `
            <section>
              <div class="sq-kicker">Step 1 of 4</div>
              <div class="sq-main-title">Get a fast repair estimate</div>
              <div class="sq-main-sub">
                Upload 1–3 clear photos of the damage. Our AI will
                scan your vehicle and estimate the repair range.
              </div>

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
                    <span>Ontario 2026 pricing</span>
                  </div>
                </div>

                <div class="sq-footer">
                  <div class="sq-footer-text">
                    Or <a href="#" id="demoLink" style="color:#60a5fa;text-decoration:none;">try a demo sample</a>
                  </div>
                  <button class="sq-btn sq-btn-primary" id="btnAnalyze">
                    Get AI estimate
                  </button>
                </div>

                <div class="sq-alert sq-alert-error" id="alertUpload"></div>
              </div>
            </section>
          `;

          const fileInput = document.getElementById("fileInput");
          const uploadZone = document.getElementById("uploadZone");
          const previewPill = document.getElementById("previewPill");
          const alertUpload = document.getElementById("alertUpload");
          const demoLink = document.getElementById("demoLink");
          const btnAnalyze = document.getElementById("btnAnalyze");

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

          fileInput.addEventListener("change",e=>{
            const files = Array.from(e.target.files).filter(f=>f.type.startsWith("image/")).slice(0,3);
            if(!files.length){
              showAlert(alertUpload,"Please select image files only.");
              return;
            }
            clearAlert(alertUpload);
            uploadedFiles = files;
            refreshPreview();
          });

          demoLink.addEventListener("click",e=>{
            e.preventDefault();
            clearAlert(alertUpload);
            showAlert(alertUpload,"Demo mode: using sample image for preview.",false);
          });

          btnAnalyze.addEventListener("click",async ()=>{
            clearAlert(alertUpload);
            if(!uploadedFiles.length){
              showAlert(alertUpload,"Please add at least one photo before continuing.");
              return;
            }
            btnAnalyze.disabled = true;
            btnAnalyze.textContent = "Analyzing…";

            try{
              const result = await callWebAnalyze();
              console.log("AI result:",result);
              if(result.error){
                showAlert(alertUpload,result.error || "Error analysing images.");
                btnAnalyze.disabled = false;
                btnAnalyze.textContent = "Get AI estimate";
                return;
              }
              aiEstimate = result;
              estimateId = result.estimate_id || (Date.now().toString());
              setStep(1);
            }catch(err){
              console.error(err);
              showAlert(alertUpload,"Something went wrong. Please try again.",true);
            }finally{
              btnAnalyze.disabled = false;
              btnAnalyze.textContent = "Get AI estimate";
            }
          });
        }

        function renderStepEstimate(){
          if(!aiEstimate){
            setStep(0);
            return;
          }
          const sev = (aiEstimate.severity || "moderate");
          const severityLabel = sev.charAt(0).toUpperCase()+sev.slice(1);
          const min = aiEstimate.estimated_min || 900;
          const max = aiEstimate.estimated_max || 1600;
          const areas = aiEstimate.areas || [];
          const types = aiEstimate.damage_types || [];
          const summary = aiEstimate.summary || "";
          const notes = aiEstimate.notes || "";

          const priceLabel = formatCurrency(min)+" – "+formatCurrency(max);

          stepContainer.innerHTML = `
            <section>
              <div class="sq-kicker">Step 2 of 4</div>
              <div class="sq-main-title">Your AI estimate</div>
              <div class="sq-main-sub">
                This is a preliminary visual estimate. Final pricing
                is confirmed after a short in-person inspection.
              </div>

              <div class="sq-summary-box">
                <div class="sq-summary-label">Severity</div>
                <div class="sq-summary-price">${severityLabel}</div>
                <div class="sq-summary-detail">Est. range: ${priceLabel}</div>
                <div class="sq-summary-badges">
                  <span class="sq-badge">Ontario 2026 prices</span>
                  <span class="sq-badge">${(types[0] || "Body & paint work")}</span>
                </div>
              </div>

              <div class="sq-summary-box">
                <div class="sq-summary-label">Areas</div>
                <div class="sq-summary-detail">
                  ${areas.length ? areas.join(", ") : "Not clearly visible from photos."}
                </div>
              </div>

              <div class="sq-summary-box">
                <div class="sq-summary-label">Summary</div>
                <div>${(summary || "").replace(/\\n/g,"<br>")}</div>
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
              <div class="sq-kicker">Step 3 of 4</div>
              <div class="sq-main-title">Tell us about you</div>
              <div class="sq-main-sub">
                Share your contact details so the shop can confirm your visit.
              </div>

              <div class="sq-stack">
                <div class="sq-field">
                  <div class="sq-label">Full name</div>
                  <input class="sq-input" id="nameInput" placeholder="Jane Doe" />
                </div>
                <div class="sq-input-row">
                  <div class="sq-field">
                    <div class="sq-label">Email</div>
                    <input class="sq-input" id="emailInput" placeholder="you@example.com" />
                  </div>
                  <div class="sq-field">
                    <div class="sq-label">Phone</div>
                    <input class="sq-input" id="phoneInput" placeholder="(555) 123-4567" />
                  </div>
                </div>
                <div class="sq-field">
                  <div class="sq-label">Vehicle</div>
                  <input class="sq-input" id="vehicleInput" placeholder="Year · Make · Model" />
                </div>
                <div class="sq-field">
                  <div class="sq-label">Notes (optional)</div>
                  <textarea class="sq-input sq-textarea" id="notesInput" placeholder="Anything the shop should know?"></textarea>
                </div>
              </div>

              <div class="sq-footer">
                <button class="sq-btn sq-btn-ghost" id="btnBackEstimate">
                  Back
                </button>
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
              showAlert(alertForm,"Please fill in all required fields.");
              return;
            }

            renderStepCalendar({
              name,
              email,
              phone,
              vehicle,
              notes:notesInput.value.trim(),
            });
          });
        }

        async function fetchAvailability(dateObj){
          const isoDate = dateObj.toISOString().slice(0,10);
          const body = {
            shop_id:shopId,
            date:isoDate
          };
          const resp = await fetch("/api/availability",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(body)
          });
          if(!resp.ok){
            return {slots:[]};
          }
          return await resp.json();
        }

        function renderStepCalendar(customerInfo){
          if(!customerInfo && window.__lastCustomerInfo){
            customerInfo = window.__lastCustomerInfo;
          }else{
            window.__lastCustomerInfo = customerInfo;
          }

          if(!customerInfo){
            setStep(2);
            return;
          }

          if(!selectedDate){
            const now = new Date();
            selectedDate = new Date(now.getFullYear(),now.getMonth(),now.getDate());
          }

          stepContainer.innerHTML = `
            <section>
              <div class="sq-kicker">Step 4 of 4</div>
              <div class="sq-main-title">Choose a time</div>
              <div class="sq-main-sub">
                Only open slots for this shop are shown. Times are local to the shop.
              </div>

              <div class="sq-calendar-shell">
                <div class="sq-calendar-header">
                  <div class="sq-calendar-title">Available times</div>
                  <div class="sq-calendar-nav">
                    <button class="sq-nav-btn" id="btnPrevDay">&#x2039;</button>
                    <button class="sq-nav-btn" id="btnNextDay">&#x203A;</button>
                  </div>
                </div>
                <div class="sq-date-line" id="dateLabel"></div>
                <div class="sq-slot-groups" id="slotGroups"></div>
                <div class="sq-slot-empty" id="slotEmpty" style="display:none;">
                  Unable to load availability. Try another date.
                </div>
              </div>

              <div class="sq-summary-box">
                <div class="sq-summary-label">Your details</div>
                <div class="sq-summary-detail">${customerInfo.name}</div>
                <div class="sq-summary-detail">${customerInfo.email} · ${customerInfo.phone}</div>
                <div class="sq-summary-detail">${customerInfo.vehicle}</div>
              </div>

              <div class="sq-footer">
                <button class="sq-btn sq-btn-ghost" id="btnBackForm">
                  Back
                </button>
                <button class="sq-btn sq-btn-primary" id="btnConfirmBooking">
                  Confirm appointment
                </button>
              </div>

              <div class="sq-alert sq-alert-error" id="alertCalendar"></div>
            </section>
          `;

          const btnPrevDay = document.getElementById("btnPrevDay");
          const btnNextDay = document.getElementById("btnNextDay");
          const dateLabel = document.getElementById("dateLabel");
          const slotGroupsContainer = document.getElementById("slotGroups");
          const slotEmpty = document.getElementById("slotEmpty");
          const alertCalendar = document.getElementById("alertCalendar");
          const btnBackForm = document.getElementById("btnBackForm");
          const confirmBookingBtn = document.getElementById("btnConfirmBooking");

          btnBackForm.addEventListener("click",()=>{
            setStep(2);
          });

          function updateDateLabel(){
            dateLabel.textContent = formatDateLabel(selectedDate);
          }

          async function loadSlots(){
            slotGroupsContainer.innerHTML = "";
            slotEmpty.style.display = "none";
            clearAlert(alertCalendar);

            let data;
            try{
              data = await fetchAvailability(selectedDate);
            }catch(e){
              console.error(e);
              slotEmpty.textContent = "Unable to load availability. Try another date.";
              slotEmpty.style.display = "block";
              return;
            }

            const slots = (data.slots || []).map(s=>({
              start:new Date(s.start),
              end:new Date(s.end),
            }));

            selectedSlot = null;

            if(!slots.length){
              slotEmpty.textContent = "No open slots for this date. Try another day.";
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
              const pill = document.createElement("button");
              pill.className = "sq-slot-pill";
              pill.textContent = label;
              pill.dataset.start = start.toISOString();
              pill.dataset.end = end.toISOString();

              pill.addEventListener("click",()=>{
                selectedSlot = {
                  startIso:pill.dataset.start,
                  endIso:pill.dataset.end,
                  label
                };
                renderSlotSelection();
              });

              if(start < now){
                pill.disabled = true;
                pill.style.opacity = .5;
                pill.style.cursor = "default";
              }

              slotGroupsContainer.appendChild(pill);
            });

            renderSlotSelection();
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

          btnPrevDay.addEventListener("click",()=>{
            const d = new Date(selectedDate);
            d.setDate(d.getDate()-1);
            selectedDate = d;
            updateDateLabel();
            loadSlots();
          });

          btnNextDay.addEventListener("click",()=>{
            const d = new Date(selectedDate);
            d.setDate(d.getDate()+1);
            selectedDate = d;
            updateDateLabel();
            loadSlots();
          });

          confirmBookingBtn.addEventListener("click",async ()=>{
            clearAlert(alertCalendar);
            if(!selectedSlot){
              showAlert(alertCalendar,"Please select a time slot before confirming.");
              return;
            }
            confirmBookingBtn.disabled = true;
            confirmBookingBtn.textContent = "Booking…";

            const payload = {
              shop_id:shopId,
              estimate_id:estimateId || "web-"+Date.now(),
              customer_name:customerInfo.name,
              phone:customerInfo.phone,
              email:customerInfo.email,
              vehicle:customerInfo.vehicle,
              notes:customerInfo.notes || "",
              slot_start:selectedSlot.startIso,
              slot_end:selectedSlot.endIso,
            };

            try{
              const res = await fetch("/api/book",{
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body:JSON.stringify(payload)
              });
              const data = await res.json();
              console.log("Booking response:",data);

              if(!data.success){
                alert("Failed to book appointment. Try again.");
                return;
              }

              alert("Appointment requested! The shop will confirm details shortly.");
              setStep(0);
            }catch(err){
              alert("Error booking appointment.");
              console.error(err);
            }

            confirmBookingBtn.disabled = false;
            confirmBookingBtn.textContent = "Confirm appointment";
          });

          updateDateLabel();
          loadSlots();
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
        except Exception:
            continue

    if not image_bytes_list:
        return {"error": "No valid images received."}

    if shop is None:
        shop = SHOPS[0]

    try:
        analysis = await run_ai_estimate_from_bytes_list(image_bytes_list, shop)
    except Exception as e:
        print("WEB ANALYZE AI ERROR:", e)
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
