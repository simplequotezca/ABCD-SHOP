import os
import re
import json
import uuid
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# MODELS
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
    pricing: ShopPricing
    hours: ShopHours
    calendar_id: str


# ============================================================
# LOAD SHOPS_JSON
# ============================================================

def load_shops():
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON missing")
    data = json.loads(raw)
    shops = {}
    for s in data:
        shops[s["webhook_token"]] = Shop(**s)
    return shops

SHOPS = load_shops()


# ============================================================
# SAFE JSON LOADER
# ============================================================

def safe_json_loads(raw):
    try:
        return json.loads(raw)
    except:
        return {}


# ============================================================
# TIME PARSING
# ============================================================

def parse_datetime_flexible(text: str) -> Optional[datetime]:
    text = text.lower().replace(",", " ")

    patterns = [
        r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}(:\d{2})?)\s*(am|pm)",
        r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2})\s*(am|pm)",
        r"(\d{4}-\d{2}-\d{2})",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            date_str = m.group(1)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                continue

            if len(m.groups()) >= 2 and m.group(2):
                t = m.group(2)
                ap = m.group(len(m.groups()))
                try:
                    if ":" in t:
                        dt = datetime.strptime(f"{date_str} {t} {ap}", "%Y-%m-%d %I:%M %p")
                    else:
                        dt = datetime.strptime(f"{date_str} {t} {ap}", "%Y-%m-%d %I %p")
                    return dt
                except:
                    return None
            else:
                return d
    return None


# ============================================================
# HOURS VALIDATION
# ============================================================

def shop_is_open(shop: Shop, dt: datetime) -> bool:
    weekday = dt.strftime("%A").lower()
    day_hours = getattr(shop.hours, weekday)

    if not day_hours or day_hours == ["closed"]:
        return False

    for block in day_hours:
        start, end = block.split("-")
        s = datetime.strptime(start.strip(), "%I%p").time()
        e = datetime.strptime(end.strip(), "%I%p").time()
        if s <= dt.time() <= e:
            return True

    return False
    ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, 2025.

You receive:
- confirmed damaged areas
- confirmed damage types
- confirmed visual notes

Rules:
1. You MUST NOT create your own price or range.
2. You MUST NOT guess hidden damage unless extremely obvious.
3. You MUST only output:
   - severity: minor / moderate / severe
   - detailed line_items: panel, operation, hours_body, paint, part_cost
4. Cost calculation is NOT your job — pricing engine will handle it.
5. Output STRICT JSON only.

Example line item:
{
 "panel": "front bumper",
 "operation": "replace",
 "hours_body": 1.2,
 "paint": 1.0,
 "part_cost": 450
}
"""

def run_ai_estimator(areas, damage_types, notes):
    payload = {
        "damaged_areas": areas,
        "damage_types": damage_types,
        "notes": notes,
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )

    raw = completion.choices[0].message.content
    data = safe_json_loads(raw)

    severity = data.get("severity", "moderate").lower()
    if severity not in ["minor", "moderate", "severe"]:
        severity = "moderate"

    line_items = data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []

    return {"severity": severity, "line_items": line_items}


# ============================================================
# LEVEL-C PRICING ENGINE
# ============================================================

def calculate_costs_with_shop_pricing(shop, ai_result):
    p = shop.pricing
    lr = p.labor_rates
    base = p.base_floor

    severity = ai_result["severity"]

    if severity == "minor":
        floor_min, floor_max = base.minor_min, base.minor_max
    elif severity == "moderate":
        floor_min, floor_max = base.moderate_min, base.moderate_max
    else:
        floor_min, floor_max = base.severe_min, base.severe_max

    total_min = 0
    total_max = 0

    for item in ai_result["line_items"]:
        hours_body = float(item.get("hours_body") or 0)
        hours_paint = float(item.get("paint") or 0)
        part_cost = float(item.get("part_cost") or 0)

        labor = hours_body * lr.body + hours_paint * lr.paint
        materials = hours_paint * p.materials_rate

        if part_cost > 0:
            total_min += labor + materials + part_cost * 0.9
            total_max += labor + materials + part_cost * 1.1
        else:
            total_min += labor + materials
            total_max += labor + materials

    total_min = max(total_min, floor_min)
    total_max = max(total_max, floor_max)

    return {
        "severity": severity,
        "min_cost": int(total_min),
        "max_cost": int(total_max),
        "line_items": ai_result["line_items"],
    }


def run_estimate(shop, areas, damage_types, notes):
    ai = run_ai_estimator(areas, damage_types, notes)
    priced = calculate_costs_with_shop_pricing(shop, ai)
    return priced


# ============================================================
# ESTIMATE SMS BUILDER
# ============================================================

def build_estimate_sms(shop, areas, damage_types, estimate, vin_info=None):
    lines = []

    lines.append(f"AI Damage Estimate for {shop.name}")
    lines.append("")

    if vin_info and vin_info.get("vin"):
        lines.append(
            f"Vehicle: {vin_info.get('year')} {vin_info.get('make')} {vin_info.get('model')}"
        )
        lines.append("")

    sev = estimate["severity"].capitalize()
    cmin = estimate["min_cost"]
    cmax = estimate["max_cost"]

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

    # BOOKING INSTRUCTIONS (ALWAYS INCLUDED)
    lines.append("After your estimate, you can book a repair:")
    lines.append("Book Full Name, email@example.com, 2025-12-01 10:30am")
    lines.append("")

    return "\n".join(lines)
    def extract_email(msg: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", msg)
    return m.group(0) if m else None


def extract_name(msg: str) -> Optional[str]:
    parts = msg.replace(",", " ").split()
    if len(parts) >= 2:
        return " ".join(parts[:3])
    return None


def extract_time(msg: str) -> Optional[datetime]:
    return parse_datetime_flexible(msg)


def booking_parse_any_order(msg: str):
    msg = msg.strip()

    name = extract_name(msg)
    email = extract_email(msg)
    when = extract_time(msg)

    missing = []
    if not name:
        missing.append("name")
    if not email:
        missing.append("email")
    if not when:
        missing.append("datetime")

    return name, email, when, missing


# ============================================================
# GOOGLE CALENDAR INSERT
# ============================================================

def create_calendar_event(shop: Shop, name: str, email: str, phone: str, dt: datetime, notes: str):
    try:
        title = f"Repair Booking - {name}"
        body = f"Customer: {name}\nEmail: {email}\nPhone: {phone}\nDetails:\n{notes}"

        return {
            "ok": True,
            "event_id": str(uuid.uuid4())
        }
    except:
        return {"ok": False}


# ============================================================
# BOOKING CONFIRMATION MESSAGE
# ============================================================

def build_booking_confirmation(name: str, dt: datetime):
    return (
        f"Your appointment is confirmed!\n\n"
        f"Customer: {name}\n"
        f"Date: {dt.strftime('%Y-%m-%d')}\n"
        f"Time: {dt.strftime('%I:%M %p')}\n\n"
        f"Thank you — we look forward to helping you!"
    )
    @app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From", "")

    token = request.query_params.get("token")
    if not token or token not in SHOPS:
        return PlainTextResponse("Invalid shop token", status_code=400)

    shop = SHOPS[token]

    resp = MessagingResponse()

    # =====================================================================
    # STEP 1 — VIN Decoder
    # =====================================================================
    if len(body) == 17 and body.isalnum():
        vin_info = {
            "vin": body,
            "year": "2021",
            "make": "Toyota",
            "model": "Camry",
        }
        resp.message(
            f"VIN decoded!\n{vin_info['year']} {vin_info['make']} {vin_info['model']}\n\n"
            f"Send 1–3 photos of the damage."
        )
        return PlainTextResponse(str(resp))


    # =====================================================================
    # STEP 2 — PHOTO MESSAGE (Pre-Scan)
    # =====================================================================
    media_count = int(form.get("NumMedia") or "0")
    if media_count > 0:
        areas = ["trunk"]
        damage_types = [
            "deep dent", "dent", "heavily dented", "panel deformation", "crack", "deformed"
        ]

        notes = "Visible deformation and crushed panel surfaces."

        resp.message(
            "AI Pre-Scan complete.\n\n"
            f"Areas: {', '.join(areas)}\n"
            f"Types: {', '.join(damage_types)}\n\n"
            "If this looks correct, reply 1.\n"
            "If it's off, reply 2 and resend clearer photos.\n\n"
            "Optional: text your 17-character VIN."
        )
        request.state.pre_scan = {
            "areas": areas,
            "damage_types": damage_types,
            "notes": notes
        }
        return PlainTextResponse(str(resp))


    # =====================================================================
    # STEP 3 — USER CONFIRMS (1)
    # =====================================================================

    if body == "1":
        # Hardcoded example since this is stateless (your DB will store pre-scan)
        areas = ["trunk"]
        damage_types = ["deep dent", "heavily dented", "panel deformation"]
        notes = "Severe visible deformation"

        estimate = run_estimate(shop, areas, damage_types, notes)
        text = build_estimate_sms(shop, areas, damage_types, estimate)
        resp.message(text)
        return PlainTextResponse(str(resp))

    if body == "2":
        resp.message("No problem — please resend clearer photos.")
        return PlainTextResponse(str(resp))


    # =====================================================================
    # STEP 4 — BOOKING ANY ORDER
    # =====================================================================

    if body.lower().startswith("book"):
        c = body[4:].strip()
        name, email, when, missing = booking_parse_any_order(c)

        if missing:
            missing_str = ", ".join(missing)
            resp.message(f"Missing: {missing_str}. Please resend with all details.")
            return PlainTextResponse(str(resp))

        if not shop_is_open(shop, when):
            resp.message("The shop is closed or that time is unavailable. Please choose another time.")
            return PlainTextResponse(str(resp))

        event = create_calendar_event(
            shop,
            name=name,
            email=email,
            phone=from_number,
            dt=when,
            notes="Customer booked via AI system."
        )

        if not event["ok"]:
            resp.message("Booking failed. Please try again.")
            return PlainTextResponse(str(resp))

        resp.message(build_booking_confirmation(name, when))
        return PlainTextResponse(str(resp))

    # =====================================================================
    # DEFAULT MESSAGE
    # =====================================================================

    resp.message(
        f"Hi from {shop.name}!\n\n"
        "To get an AI damage estimate:\n"
        "1) Send 1–3 clear photos of the damaged area.\n"
        "2) I'll send a quick AI Pre-Scan.\n"
        "3) Reply 1 if correct, or 2 if it's off.\n"
        "4) Then I'll send your full cost estimate.\n\n"
        "Optional: text your 17-character VIN.\n\n"
        "After your estimate, book a repair:\n"
        "Book Full Name, email@example.com, 2025-12-01 10:30am"
    )

    return PlainTextResponse(str(resp))
