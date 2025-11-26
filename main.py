# ============================================================
# AI Estimator – Legacy OpenAI SDK Compatible (Final Version)
# ============================================================

import os
import re
import json
import uuid
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# ============================================================
# UNIVERSAL OPENAI CLIENT (WORKS WITH ANY VERSION)
# ============================================================

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# TWILIO FALLBACK
# ============================================================

try:
    from twilio.twiml.messaging_response import MessagingResponse
except ImportError:
    class MessagingResponse:
        def __init__(self):
            self.msg = []
        def message(self, t):
            self.msg.append(t)
        def __str__(self):
            return "<Response>" + "".join(f"<Message>{m}</Message>" for m in self.msg) + "</Response>"

app = FastAPI()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")


# ============================================================
# SHOP MODELS
# ============================================================

class LaborRates(BaseModel):
    body: float
    paint: float

class BaseFloor(BaseModel):
    minor_min: int
    minor_max: int
    moderate_min: int
    moderate_max: int
    severe_min: int
    severe_max: int

class ShopPricing(BaseModel):
    labor_rates: LaborRates
    materials_rate: float
    base_floor: BaseFloor

class ShopHours(BaseModel):
    monday: List[str]
    tuesday: List[str]
    wednesday: List[str]
    thursday: List[str]
    friday: List[str]
    saturday: List[str]
    sunday: List[str]

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: ShopPricing
    hours: ShopHours


# ============================================================
# LOAD SHOPS
# ============================================================

def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON missing")
    data = json.loads(raw)
    out = {}
    for s in data:
        shop = Shop(
            id=s["id"],
            name=s["name"],
            webhook_token=s["webhook_token"],
            calendar_id=s["calendar_id"],
            pricing=ShopPricing(**s["pricing"]),
            hours=ShopHours(**s["hours"])
        )
        out[shop.webhook_token] = shop
    return out

SHOPS_BY_TOKEN = load_shops()

# ============================================================
# SESSION HANDLING
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_MINUTES = 120

def sess_key(shop: Shop, phone: str):
    return f"{shop.id}:{phone}"

def get_session(shop: Shop, phone: str):
    key = sess_key(shop, phone)
    now = datetime.utcnow()

    s = SESSIONS.get(key)
    if s:
        try:
            created = datetime.fromisoformat(s["_created"])
        except:
            created = now
        if now - created > timedelta(minutes=SESSION_TTL_MINUTES):
            s = None

    if not s:
        s = {"_created": now.isoformat()}
        SESSIONS[key] = s

    return s


# ============================================================
# UTILITIES
# ============================================================

def safe_json(raw):
    try:
        return json.loads(raw)
    except:
        return {}

def download_media(url: str) -> bytes:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    else:
        r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content

def to_data_url(data: bytes, ctype="image/jpeg"):
    return f"data:{ctype};base64,{base64.b64encode(data).decode()}"


# ============================================================
# DATE/TIME PARSING
# (unchanged — fully stable)
# ============================================================

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12
}

def parse_time_any(text: str):
    t = text.lower()

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or "0")
        ap = m.group(3)
        if ap == "pm" and h != 12: h += 12
        if ap == "am" and h == 12: h = 0
        return h, mi

    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return h, mi

    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        mi = 0
        ap = m.group(2)
        if ap == "pm" and h != 12: h += 12
        if ap == "am" and h == 12: h = 0
        return h, mi

    return None


def parse_date_any(text: str):
    t = text.lower().replace(",", " ")
    t = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", t)

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else datetime.utcnow().year
        if year < 100: year += 2000
        return year, month, day

    m = re.search(r"\b([a-z]{3,9})\s+(\d{1,2})\b", t)
    if m:
        w = m.group(1)
        d = int(m.group(2))
        month = MONTHS.get(w[:3], MONTHS.get(w))
        if month:
            return datetime.utcnow().year, month, d

    m = re.search(r"\b(\d{1,2})\s+([a-z]{3,9})\b", t)
    if m:
        d = int(m.group(1))
        w = m.group(2)
        month = MONTHS.get(w[:3], MONTHS.get(w))
        if month:
            return datetime.utcnow().year, month, d

    return None


def parse_datetime_any(text: str):
    missing = []
    date_info = parse_date_any(text)
    time_info = parse_time_any(text)

    if not date_info: missing.append("date")
    if not time_info: missing.append("time")

    if date_info and time_info:
        y, m, d = date_info
        h, mi = time_info
        try:
            return datetime(y, m, d, h, mi), []
        except:
            return None, ["date"]

    return None, missing


# ============================================================
# VIN DECODER
# ============================================================

VIN_PROMPT = """
You decode 17-character VINs. Respond ONLY in JSON with keys:
year, make, model, body_style
""".strip()

def decode_vin(vin: str):
    c = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": VIN_PROMPT},
            {"role": "user", "content": vin}
        ]
    )
    out = safe_json(c["choices"][0]["message"]["content"])
    out["vin"] = vin
    return out


# ============================================================
# PRE-SCAN (MULTI-IMAGE FUSION)
# ============================================================

PRESCAN_PROMPT = """
Fuse all photos and identify:
- areas
- damage_types
- notes
Respond ONLY in JSON.
""".strip()

def run_prescan(images: List[str]):
    # Legacy ChatCompletion format
    messages = [
        {"role": "system", "content": PRESCAN_PROMPT},
        {"role": "user", "content": "Analyze these images."}
    ]

    # Provide up to 3 images
    for url in images[:3]:
        messages.append({
            "role": "user",
            "content": f"[IMAGE]{url}"
        })

    c = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    data = safe_json(c["choices"][0]["message"]["content"])
    areas = [a.lower() for a in data.get("areas", [])]
    dmg = [d.lower() for d in data.get("damage_types", [])]
    notes = data.get("notes", "")
    if isinstance(notes, list):
        notes = " ".join(notes)

    return {"areas": areas, "damage_types": dmg, "notes": notes}


# ============================================================
# DAMAGE ESTIMATOR
# ============================================================

ESTIMATOR_PROMPT = """
Output severity, summary, and line_items in JSON.
No dollar signs. No totals.
""".strip()

def run_ai_estimator(areas, dmg, notes):
    payload = {"areas": areas, "damage_types": dmg, "notes": notes}

    c = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": ESTIMATOR_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ]
    )

    data = safe_json(c["choices"][0]["message"]["content"])
    sev = data.get("severity", "unknown").lower()
    if sev not in ["minor", "moderate", "severe"]:
        sev = "unknown"

    items = data.get("line_items", [])
    if not isinstance(items, list):
        items = []

    return {
        "severity": sev,
        "summary": data.get("summary", ""),
        "line_items": items
    }


# ============================================================
# PRICING ENGINE
# ============================================================

def price_with_shop(shop: Shop, ai):
    sev = ai["severity"]
    base = shop.pricing.base_floor

    if sev == "minor": floor_min, floor_max = base.minor_min, base.minor_max
    elif sev == "moderate": floor_min, floor_max = base.moderate_min, base.moderate_max
    elif sev == "severe": floor_min, floor_max = base.severe_min, base.severe_max
    else: floor_min, floor_max = 400, 900

    lr = shop.pricing.labor_rates
    mat = shop.pricing.materials_rate

    tmin = 0.0
    tmax = 0.0

    for it in ai["line_items"]:
        hb = float(it.get("hours_body") or 0)
        hp = float(it.get("hours_paint") or 0)
        pc = float(it.get("part_cost") or 0)

        labor = hb * lr.body + hp * lr.paint
        materials = hp * mat

        if pc > 0:
            tmin += labor + materials + pc * 0.9
            tmax += labor + materials + pc * 1.1
        else:
            tmin += labor + materials
            tmax += labor + materials

    tmin = max(tmin, floor_min)
    tmax = max(tmax, floor_max)

    return {
        "severity": sev,
        "summary": ai["summary"],
        "min_cost": int(tmin),
        "max_cost": int(tmax),
        "line_items": ai["line_items"]
    }


def run_estimate(shop, areas, dmg, notes):
    ai = run_ai_estimator(areas, dmg, notes)
    return price_with_shop(shop, ai)


def build_estimate_sms(shop, areas, dmg, est, vin=None):
    L = []
    L.append(f"AI Damage Estimate for {shop.name}\n")

    if vin and vin.get("vin"):
        L.append(
            f"Vehicle: {vin.get('year','?')} {vin.get('make','?')} {vin.get('model','?')}\n"
        )

    L.append(f"Severity: {est['severity'].capitalize()}")
    L.append(f"Estimated Cost (Ontario 2025): ${est['min_cost']} – ${est['max_cost']}\n")

    if areas:
        L.append("Areas:")
        L.append("- " + ", ".join(areas) + "\n")
    if dmg:
        L.append("Damage Types:")
        L.append("- " + ", ".join(dmg) + "\n")
    if est["summary"]:
        L.append(est["summary"] + "\n")

    L.append("To book a repair, reply with:")
    L.append("Book John Doe, john@example.com, Nov 29 2pm\n")
    return "\n".join(L)


# ============================================================
# BOOKING PARSER
# ============================================================

def extract_email(text: str):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None

def extract_name(text: str, email: Optional[str]):
    t = text
    if email: t = t.replace(email, " ")
    t = re.sub(r"\bbook\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " ", t)
    t = re.sub(r"\d{1,2}/\d{1,2}(/\d{2,4})?", " ", t)
    t = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
               " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\d{1,2}(:\d{2})?\s*(am|pm)", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    if not t: return None
    return " ".join(t.split()[:3])

def parse_booking_any_order(raw: str):
    email = extract_email(raw)
    dt, missing_dt = parse_datetime_any(raw)
    name = extract_name(raw, email)

    missing = []
    if not name: missing.append("name")
    if not email: missing.append("email")
    if dt is None: missing.extend(missing_dt)

    missing = list(dict.fromkeys(missing))
    return name, email, dt, missing


# ============================================================
# GOOGLE CALENDAR STUB
# ============================================================

def create_calendar_event(shop, name, email, phone, dt, notes):
    return {"ok": True, "event_id": str(uuid.uuid4())}


def build_booking_confirmation(name, dt):
    return (
        f"Your appointment is confirmed!\n\n"
        f"Name: {name}\n"
        f"Date: {dt.strftime('%Y-%m-%d')}\n"
        f"Time: {dt.strftime('%I:%M %p')}\n\n"
        f"Thank you — we look forward to helping you!"
    )


# ============================================================
# TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""

    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token or "")
    if not shop:
        return PlainTextResponse("Invalid shop token", status_code=400)

    session = get_session(shop, from_number)
    reply = MessagingResponse()

    # VIN
    vin_candidate = body.replace(" ", "").upper()
    if len(vin_candidate) == 17 and vin_candidate.isalnum():
        vin_info = decode_vin(vin_candidate)
        session["vin"] = vin_info
        reply.message(
            f"VIN decoded: {vin_info.get('year','?')} "
            f"{vin_info.get('make','?')} {vin_info.get('model','?')}.\n\n"
            "Now send 1–3 clear photos of the damage."
        )
        return PlainTextResponse(str(reply), media_type="application/xml")

    # PHOTOS
    media_count = int(form.get("NumMedia") or "0")
    if media_count > 0:
        imgs = []
        for i in range(media_count):
            url = form.get(f"MediaUrl{i}")
            ctype = form.get(f"MediaContentType{i}") or "image/jpeg"
            if url:
                try:
                    data = download_media(url)
                    imgs.append(to_data_url(data, ctype))
                except:
                    pass

        if not imgs:
            reply.message("I couldn't read the photos. Try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        prescan = run_prescan(imgs)
        session["prescan"] = prescan

        L = []
        L.append(f"AI Pre-Scan for {shop.name}\n")
        if prescan["areas"]:
            L.append("Visible areas:")
            L.append("- " + ", ".join(prescan["areas"]) + "\n")
        if prescan["damage_types"]:
            L.append("Damage types:")
            L.append("- " + ", ".join(prescan["damage_types"]) + "\n")
        if prescan["notes"]:
            L.append("Notes:")
            L.append(prescan["notes"] + "\n")
        L.append("If this looks correct, reply 1.")
        L.append("If not, reply 2 and send clearer photos.")

        reply.message("\n".join(L))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # CONFIRM 1 → estimate
    if body == "1" and "prescan" in session:
        ps = session["prescan"]
        areas, dmg, notes = ps["areas"], ps["damage_types"], ps["notes"]
        est = run_estimate(shop, areas, dmg, notes)
        session["estimate"] = est

        vin = session.get("vin")
        reply.message(build_estimate_sms(shop, areas, dmg, est, vin))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # REJECT 2 → new photos
    if body == "2" and "prescan" in session:
        session.pop("prescan", None)
        reply.message("No problem — send clearer photos.")
        return PlainTextResponse(str(reply), media_type="application/xml")

    # BOOKING
    lower = body.lower()
    looks_booking = False

    if lower.startswith("book"):
        looks_booking = True
    elif extract_email(body):
        if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", lower) or \
           re.search(r"\d{4}-\d{2}-\d{2}", lower) or \
           re.search(r"\d{1,2}/\d{1,2}", lower):
            looks_booking = True

    if looks_booking:
        if "estimate" not in session:
            reply.message("Please send photos first to get an estimate.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        name, email, dt, missing = parse_booking_any_order(body)

        if missing:
            reply.message(
                "Missing booking info.\n"
                "Include:\n- Full name\n- Email\n- Date + time\n\n"
                "Example:\n"
                "Book John Doe, john@example.com, Nov 29 2pm"
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        if not shop_open(shop, dt):
            reply.message("That time is outside shop hours. Try another.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        event = create_calendar_event(shop, name, email, from_number, dt, "AI booking")
        if not event.get("ok"):
            reply.message("Booking error — try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        reply.message(build_booking_confirmation(name, dt))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # default intro
    intro = []
    intro.append(f"Hi from {shop.name}!\n")
    intro.append("To get an AI damage estimate:")
    intro.append("1) Send 1–3 photos of the damage.")
    intro.append("2) Confirm the pre-scan with 1.")
    intro.append("3) Receive your estimate.\n")
    intro.append("Optional:")
    intro.append("- Send VIN to decode.")
    intro.append("- Book with: Book John Doe, john@example.com, Nov 29 2pm")

    reply.message("\n".join(intro))
    return PlainTextResponse(str(reply), media_type="application/xml")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Estimator Running"}
