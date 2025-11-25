import os
import re
import json
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

# ============================================================
# Environment and FastAPI setup
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

app = FastAPI()

# ============================================================
# Data models
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
# Shop config loader (Option B architecture)
# ============================================================

def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON env var is required")

    data = json.loads(raw)
    by_token: Dict[str, Shop] = {}
    for item in data:
        shop = Shop(
            id=item["id"],
            name=item["name"],
            webhook_token=item["webhook_token"],
            calendar_id=item["calendar_id"],
            pricing=ShopPricing(**item["pricing"]),
            hours=ShopHours(**item["hours"]),
        )
        by_token[shop.webhook_token] = shop
    return by_token

SHOPS_BY_TOKEN = load_shops()

# ============================================================
# In-memory session store (per shop + phone)
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_MINUTES = 120

def session_key(shop: Shop, phone: str) -> str:
    return f"{shop.id}:{phone}"

def get_session(shop: Shop, phone: str) -> Dict[str, Any]:
    key = session_key(shop, phone)
    now = datetime.utcnow()
    session = SESSIONS.get(key)
    if session:
        created = session.get("_created_at", now)
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except Exception:
                created = now
        if now - created > timedelta(minutes=SESSION_TTL_MINUTES):
            session = None
    if not session:
        session = {"_created_at": datetime.utcnow().isoformat()}
        SESSIONS[key] = session
    return session

# ============================================================
# Utility helpers
# ============================================================

def safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {}

def bytes_to_data_url(data: bytes, content_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{content_type};base64,{b64}"

def download_media(url: str) -> bytes:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    else:
        resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content

# ============================================================
# Time parsing and shop hours
# ============================================================

def parse_datetime_flexible(text: str) -> Optional[datetime]:
    """
    Accepts formats like:
      2025-11-28 3pm
      2025-11-28 15:30
      2025-11-28 3:30 pm
    Returns a timezone-naive datetime (assumed local).
    """
    text = text.strip()
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text)

    # YYYY-MM-DD HH:MM am/pm
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.IGNORECASE)
    if m:
        date_str = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3) or "0")
        ap = m.group(4).lower()
        if ap == "pm" and hour != 12:
            hour += 12
        if ap == "am" and hour == 12:
            hour = 0
        try:
            return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
        except Exception:
            return None

    # YYYY-MM-DD HH:MM (24h)
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})", text)
    if m:
        date_str = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3))
        try:
            return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
        except Exception:
            return None

    return None

def shop_is_open(shop: Shop, dt: datetime) -> bool:
    weekday = dt.strftime("%A").lower()
    day_hours: List[str] = getattr(shop.hours, weekday)
    if not day_hours or day_hours == ["closed"]:
        return False

    for block in day_hours:
        try:
            start_str, end_str = block.split("-")
            start_str = start_str.strip().lower()
            end_str = end_str.strip().lower()

            def parse_time(t: str):
                m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
                if not m:
                    return None
                hour = int(m.group(1))
                minute = int(m.group(2) or "0")
                ap = m.group(3).lower()
                if ap == "pm" and hour != 12:
                    hour += 12
                if ap == "am" and hour == 12:
                    hour = 0
                return hour, minute

            parsed_start = parse_time(start_str)
            parsed_end = parse_time(end_str)
            if not parsed_start or not parsed_end:
                continue

            sh, sm = parsed_start
            eh, em = parsed_end

            start_dt = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end_dt = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
            if start_dt <= dt <= end_dt:
                return True
        except Exception:
            continue

    return False

# ============================================================
# VIN decoding (simple heuristic via OpenAI)
# ============================================================

VIN_SYSTEM_PROMPT = """
You are a VIN decoding assistant.

You receive a single 17-character VIN string.
Your job:
- Decode the VIN into vehicle year, make, model, and body style IF POSSIBLE.
- If you are not at least 90% sure, set any unknown field to "unknown".
- NEVER invent trim levels or package names.
- Only include: year, make, model, body_style.

Respond in strict JSON:

{
  "year": "YYYY or unknown",
  "make": "Ford or unknown",
  "model": "F-150 or unknown",
  "body_style": "sedan / SUV / truck / coupe / unknown"
}
""".strip()

def decode_vin_with_ai(vin: str) -> Dict[str, str]:
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VIN_SYSTEM_PROMPT},
            {"role": "user", "content": vin},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)
    data["vin"] = vin
    return data

# ============================================================
# Pre-scan multi-image fusion (vision)
# ============================================================

PRE_SCAN_SYSTEM_PROMPT = """
You are an AI damage triage assistant for an auto body shop.

You will receive 1–3 photos of vehicle damage.

Your job:
1) Look across ALL photos (multi-image fusion).
2) Identify the main exterior areas damaged (e.g. "front bumper", "hood", "left fender", "trunk", "rear bumper").
3) Identify high-level damage types (e.g. "scratch", "dent", "deep dent", "crack", "panel deformation", "heavily dented").
4) Write 2–4 bullet-point NOTES describing what you see in plain language.
5) DO NOT mention cost, prices, or repair methods.

Respond in this JSON schema only:

{
  "areas": ["front bumper", "hood"],
  "damage_types": ["deep dent", "panel deformation"],
  "notes": [
    "Front bumper is pushed inwards on the right side.",
    "Hood has a large crease above the grille."
  ]
}
""".strip()

def run_pre_scan(image_data_urls: List[str]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Customer photos for collision pre-scan. "
                "Fuse observations across ALL photos and respond ONLY with the JSON schema."
            ),
        }
    ]
    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PRE_SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)
    areas = data.get("areas") or []
    damage_types = data.get("damage_types") or []
    notes_list = data.get("notes") or []
    if isinstance(notes_list, list):
        notes = " ".join(str(n) for n in notes_list)
    else:
        notes = str(notes_list)
    return {
        "areas": [a.strip().lower() for a in areas],
        "damage_types": [d.strip().lower() for d in damage_types],
        "notes": notes.strip(),
    }

# ============================================================
# Level-C estimator (AI severity + line items; pricing by shop)
# ============================================================

ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, Canada (year 2025).

You will receive:
- confirmed damaged areas
- confirmed damage types
- short visual notes

Rules:
1) You MUST NOT output any cost or price numbers.
2) You MUST NOT guess hidden damage that is not visually evident.
3) Your job is only to:
   - choose severity: minor / moderate / severe
   - produce a short summary
   - produce a list of line_items with panel, operation, hours_body, hours_paint, and part_cost (part_cost may be 0 if unknown).

Use this JSON schema only:

{
  "severity": "minor | moderate | severe | unknown",
  "summary": "2–4 sentence explanation.",
  "line_items": [
    {
      "panel": "front bumper",
      "operation": "repair | replace | R&I | R&R | refinish | blend",
      "hours_body": 1.5,
      "hours_paint": 1.0,
      "part_cost": 0
    }
  ]
}
""".strip()

def run_ai_estimator(areas: List[str], damage_types: List[str], notes: str) -> Dict[str, Any]:
    payload = {
        "areas": areas,
        "damage_types": damage_types,
        "notes": notes,
    }
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)
    severity = (data.get("severity") or "unknown").lower()
    if severity not in {"minor", "moderate", "severe"}:
        severity = "unknown"
    line_items = data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []
    summary = data.get("summary") or ""
    return {
        "severity": severity,
        "line_items": line_items,
        "summary": summary,
    }

def calculate_costs_with_shop_pricing(shop: Shop, ai_result: Dict[str, Any]) -> Dict[str, Any]:
    severity = ai_result["severity"]
    pricing = shop.pricing
    lr = pricing.labor_rates
    base = pricing.base_floor

    if severity == "minor":
        floor_min, floor_max = base.minor_min, base.minor_max
    elif severity == "moderate":
        floor_min, floor_max = base.moderate_min, base.moderate_max
    elif severity == "severe":
        floor_min, floor_max = base.severe_min, base.severe_max
    else:
        floor_min, floor_max = 400, 900

    total_min = 0.0
    total_max = 0.0

    for item in ai_result["line_items"]:
        hours_body = float(item.get("hours_body") or 0)
        hours_paint = float(item.get("hours_paint") or 0)
        part_cost = float(item.get("part_cost") or 0)

        labor = hours_body * lr.body + hours_paint * lr.paint
        materials = hours_paint * pricing.materials_rate

        if part_cost > 0:
            total_min += labor + materials + part_cost * 0.9
            total_max += labor + materials + part_cost * 1.1
        else:
            total_min += labor + materials
            total_max += labor + materials

    total_min = max(total_min, float(floor_min))
    total_max = max(total_max, float(floor_max))

    return {
        "severity": severity,
        "min_cost": int(total_min),
        "max_cost": int(total_max),
        "summary": ai_result["summary"],
        "line_items": ai_result["line_items"],
    }

def run_estimate(shop: Shop, areas: List[str], damage_types: List[str], notes: str) -> Dict[str, Any]:
    ai_result = run_ai_estimator(areas, damage_types, notes)
    priced = calculate_costs_with_shop_pricing(shop, ai_result)
    return priced

# ============================================================
# Estimate SMS builder (with booking instructions)
# ============================================================

def build_estimate_sms(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
    vin_info: Optional[Dict[str, str]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"AI Damage Estimate for {shop.name}")
    lines.append("")

    if vin_info and vin_info.get("vin"):
        yr = vin_info.get("year", "unknown")
        mk = vin_info.get("make", "unknown")
        md = vin_info.get("model", "unknown")
        lines.append(f"Vehicle: {yr} {mk} {md}")
        lines.append("")

    sev = estimate.get("severity", "unknown").capitalize()
    cmin = estimate.get("min_cost", 0)
    cmax = estimate.get("max_cost", 0)
    summary = estimate.get("summary", "")

    lines.append(f"Severity: {sev}")
    lines.append(f"Estimated Cost (Ontario 2025): ${cmin} – ${cmax}")
    lines.append("")

    if areas:
        lines.append("Areas:")
        lines.append("- " + ", ".join(areas))
        lines.append("")
    if damage_types:
        lines.append("Damage Types:")
        lines.append("- " + ", ".join(damage_types))
        lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")

    lines.append("This is a visual pre-estimate only. Final pricing may change after in-person inspection.")
    lines.append("")

    lines.append("After your estimate, you can book a repair by replying:")
    lines.append("Book Full Name, email@example.com, 2025-12-01 10:30am")
    lines.append("")

    return "\n".join(lines)

# ============================================================
# Booking parsing (any order: name, email, datetime)
# ============================================================

def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None

def extract_datetime_anywhere(text: str) -> Optional[datetime]:
    return parse_datetime_flexible(text)

def extract_name_guess(text: str, email: Optional[str]) -> Optional[str]:
    # remove email and date from text
    if email:
        text = text.replace(email, " ")
    # remove obvious date-like fragments (rough)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;")
    if not text:
        return None
    # keep first 3 words as name
    parts = text.split()
    return " ".join(parts[:3])

def parse_booking_any_order(raw: str):
    text = raw.strip()
    if text.lower().startswith("book"):
        text = text[4:].strip(" ,;")

    email = extract_email(text)
    when = extract_datetime_anywhere(text)
    name = extract_name_guess(text, email)

    missing = []
    if not name:
        missing.append("name")
    if not when:
        missing.append("date/time (YYYY-MM-DD 3pm)")
    if not email:
        missing.append("email")

    return name, email, when, missing

# ============================================================
# Google Calendar stub (you can replace with real API later)
# ============================================================

def create_calendar_event_stub(shop: Shop, name: str, email: str, phone: str, dt: datetime, notes: str) -> Dict[str, Any]:
    # This is a stub; replace with google-api-python-client integration if desired.
    event_id = str(uuid.uuid4())
    return {"ok": True, "event_id": event_id}

def build_booking_confirmation(name: str, dt: datetime) -> str:
    return (
        f"Your appointment is confirmed!\n\n"
        f"Name: {name}\n"
        f"Date: {dt.strftime('%Y-%m-%d')}\n"
        f"Time: {dt.strftime('%I:%M %p')}\n\n"
        f"Thank you – we look forward to helping you."
    )

# ============================================================
# Main SMS webhook
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

    # 1) VIN decoding
    vin_candidate = body.replace(" ", "").upper()
    if len(vin_candidate) == 17 and vin_candidate.isalnum():
        vin_info = decode_vin_with_ai(vin_candidate)
        session["vin_info"] = vin_info
        reply.message(
            f"VIN decoded: {vin_info.get('year', 'unknown')} {vin_info.get('make', 'unknown')} {vin_info.get('model', 'unknown')}."
            "\n\nSend 1–3 clear photos of the damage to continue your AI estimate."
        )
        return PlainTextResponse(str(reply), media_type="application/xml")

    # 2) Photos => Pre-scan
    num_media = int(form.get("NumMedia") or "0")
    if num_media > 0:
        image_data_urls: List[str] = []
        for idx in range(num_media):
            url = form.get(f"MediaUrl{idx}")
            if not url:
                continue
            try:
                data = download_media(url)
                data_url = bytes_to_data_url(data)
                image_data_urls.append(data_url)
            except Exception:
                continue

        if not image_data_urls:
            reply.message("I couldn't read the photos. Please try sending them again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        try:
            pre_scan = run_pre_scan(image_data_urls)
        except Exception:
            reply.message("Sorry — there was an error analyzing the photos. Please try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        areas = pre_scan.get("areas", [])
        damage_types = pre_scan.get("damage_types", [])
        notes = pre_scan.get("notes", "")

        session["pre_scan"] = {
            "areas": areas,
            "damage_types": damage_types,
            "notes": notes,
        }

        lines: List[str] = []
        lines.append(f"AI Pre-Scan for {shop.name}")
        lines.append("")
        if areas:
            lines.append("Here's what I can clearly see from your photo(s):")
            lines.append("- " + ", ".join(areas))
            lines.append("")
        if damage_types:
            lines.append("Damage types I can see:")
            lines.append("- " + ", ".join(damage_types))
            lines.append("")
        if notes:
            lines.append("Notes:")
            lines.append(notes)
            lines.append("")
        lines.append("If this looks roughly correct, reply 1 and I'll send a full estimate with cost.")
        lines.append("If it's off, reply 2 and you can send clearer / wider photos.")
        lines.append("")
        lines.append("Optional: you can also text your 17-character VIN to decode your vehicle details.")

        reply.message("\n".join(lines))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # 3) User confirms or rejects pre-scan
    if body.strip() == "1" and "pre_scan" in session:
        pre = session["pre_scan"]
        areas = pre.get("areas", [])
        damage_types = pre.get("damage_types", [])
        notes = pre.get("notes", "")

        estimate = run_estimate(shop, areas, damage_types, notes)
        session["last_estimate"] = estimate

        vin_info = session.get("vin_info")
        sms_text = build_estimate_sms(shop, areas, damage_types, estimate, vin_info=vin_info)
        reply.message(sms_text)
        return PlainTextResponse(str(reply), media_type="application/xml")

    if body.strip() == "2" and "pre_scan" in session:
        session.pop("pre_scan", None)
        reply.message("No problem — please send clearer / wider photos of the damage.")
        return PlainTextResponse(str(reply), media_type="application/xml")

    # 4) Booking flow
    if body.lower().startswith("book"):
        name, email, when, missing = parse_booking_any_order(body)
        if missing:
            reply.message(
                "I couldn't read that fully. Please include: name, email, and date/time in this format:"
                "\nBook Full Name, email@example.com, 2025-12-01 10:30am"
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        if not shop_is_open(shop, when):
            reply.message(
                "That time appears to be outside shop hours or unavailable. "
                "Please choose another date/time within business hours."
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        event = create_calendar_event_stub(
            shop=shop,
            name=name,
            email=email,
            phone=from_number,
            dt=when,
            notes="AI estimate booking",
        )
        if not event.get("ok"):
            reply.message("There was a problem creating your booking. Please try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        reply.message(build_booking_confirmation(name, when))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # 5) Default onboarding message
    intro_lines: List[str] = []
    intro_lines.append(f"Hi from {shop.name}!")
    intro_lines.append("")
    intro_lines.append("To get an AI-powered damage estimate:")
    intro_lines.append("1) Send 1–3 clear photos of the damaged area.")
    intro_lines.append("2) I'll send a quick AI Pre-Scan.")
    intro_lines.append("3) Reply 1 if it looks right, or 2 if it's off.")
    intro_lines.append("4) Then I'll send your full Ontario 2025 cost estimate.")
    intro_lines.append("")
    intro_lines.append("Optional:")
    intro_lines.append("- Text your 17-character VIN to decode your vehicle details.")
    intro_lines.append("- After your estimate, book a repair by replying:")
    intro_lines.append("  Book Full Name, email@example.com, 2025-12-01 10:30am")

    reply.message("\n".join(intro_lines))
    return PlainTextResponse(str(reply), media_type="application/xml")
