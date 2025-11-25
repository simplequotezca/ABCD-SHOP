import os
import json
import re
import base64
import uuid
import traceback
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

from dateutil import parser as date_parser

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from contextlib import contextmanager

# ============================================================
# DB SETUP (PostgreSQL)
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required for PostgreSQL database")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=False, index=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicles = relationship("Vehicle", back_populates="customer")
    estimates = relationship("Estimate", back_populates="customer")
    bookings = relationship("Booking", back_populates="customer")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    vin = Column(String, nullable=True)
    year = Column(String, nullable=True)
    make = Column(String, nullable=True)
    model = Column(String, nullable=True)
    body_style = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="vehicles")


class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    shop_id = Column(String, nullable=False)
    areas = Column(JSON, nullable=True)
    damage_types = Column(JSON, nullable=True)
    severity = Column(String, nullable=True)
    min_cost = Column(Integer, nullable=True)
    max_cost = Column(Integer, nullable=True)
    line_items = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="estimates")


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    shop_id = Column(String, nullable=False)
    appointment_time = Column(DateTime, nullable=False)
    calendar_event_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="bookings")


Base.metadata.create_all(bind=engine)


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_or_create_customer(
    db, phone: str, full_name: Optional[str] = None, email: Optional[str] = None
) -> Customer:
    customer = db.query(Customer).filter(Customer.phone == phone).first()
    if customer:
        updated = False
        if full_name and not customer.full_name:
            customer.full_name = full_name
            updated = True
        if email and not customer.email:
            customer.email = email
            updated = True
        if updated:
            db.add(customer)
        return customer

    customer = Customer(full_name=full_name, phone=phone, email=email)
    db.add(customer)
    db.flush()
    return customer


def upsert_vehicle_from_vin(
    db, customer: Customer, vin_info: Dict[str, str]
) -> Optional[Vehicle]:
    vin = vin_info.get("vin")
    if not vin:
        return None

    vehicle = (
        db.query(Vehicle)
        .filter(Vehicle.customer_id == customer.id, Vehicle.vin == vin)
        .first()
    )
    if not vehicle:
        vehicle = Vehicle(
            customer_id=customer.id,
            vin=vin,
            year=vin_info.get("year"),
            make=vin_info.get("make"),
            model=vin_info.get("model"),
            body_style=vin_info.get("body_style"),
        )
        db.add(vehicle)
        db.flush()
    else:
        changed = False
        for field in ["year", "make", "model", "body_style"]:
            val = vin_info.get(field)
            if val and getattr(vehicle, field) != val:
                setattr(vehicle, field, val)
                changed = True
        if changed:
            db.add(vehicle)
    return vehicle


def save_estimate_record(
    db,
    customer: Customer,
    shop_id: str,
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
    notes: str,
) -> Estimate:
    est = Estimate(
        customer_id=customer.id,
        shop_id=shop_id,
        areas=areas or [],
        damage_types=damage_types or [],
        severity=estimate.get("severity"),
        min_cost=int(estimate.get("min_cost") or 0),
        max_cost=int(estimate.get("max_cost") or 0),
        line_items=estimate.get("line_items") or [],
        notes=notes or "",
    )
    db.add(est)
    db.flush()
    return est


def save_booking_record(
    db,
    customer: Customer,
    shop_id: str,
    appointment_time: datetime,
    calendar_event_id: Optional[str],
) -> Booking:
    booking = Booking(
        customer_id=customer.id,
        shop_id=shop_id,
        appointment_time=appointment_time,
        calendar_event_id=calendar_event_id,
    )
    db.add(booking)
    db.flush()
    return booking


# ============================================================
# FASTAPI + OPENAI CLIENT
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# SHOP CONFIG / PRICING
# ============================================================

class LaborRates(BaseModel):
    body: float
    paint: float
    frame: Optional[float] = None


class BaseFloor(BaseModel):
    minor_min: float
    minor_max: float
    moderate_min: float
    moderate_max: float
    severe_min: float
    severe_max: float


class ShopPricing(BaseModel):
    labor_rates: LaborRates
    materials_rate: float = 30.0
    blend_multiplier: float = 0.5
    base_floor: BaseFloor


DEFAULT_HOURS: Dict[str, List[str]] = {
    "mon": ["09:00", "17:00"],
    "tue": ["09:00", "17:00"],
    "wed": ["09:00", "19:00"],
    "thu": ["09:00", "19:00"],
    "fri": ["09:00", "19:00"],
    "sat": ["09:00", "17:00"],
    "sun": [],
}


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    pricing: Optional[ShopPricing] = None
    hours: Optional[Dict[str, List[str]]] = None


def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        default = Shop(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
            hours=DEFAULT_HOURS,
        )
        return {default.webhook_token: default}

    data = json.loads(raw)
    by_token: Dict[str, Shop] = {}
    for item in data:
        pricing_obj = None
        if "pricing" in item and isinstance(item["pricing"], dict):
            try:
                pricing_obj = ShopPricing(**item["pricing"])
            except Exception as e:
                print("Error parsing pricing for shop:", item.get("id"), repr(e))
                pricing_obj = None

        hours = item.get("hours") or DEFAULT_HOURS

        shop = Shop(
            id=item.get("id", "unknown"),
            name=item.get("name", "Auto Body Shop"),
            webhook_token=item["webhook_token"],
            calendar_id=item.get("calendar_id"),
            pricing=pricing_obj,
            hours=hours,
        )
        by_token[shop.webhook_token] = shop
    return by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()

# ============================================================
# IN-MEMORY SESSION STORE
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_MINUTES = 120


def session_key(shop: Shop, phone: str) -> str:
    return f"{shop.id}:{phone}"


def cleanup_sessions() -> None:
    now = datetime.utcnow()
    expired = []
    for k, v in SESSIONS.items():
        created_at = v.get("created_at")
        if isinstance(created_at, datetime) and now - created_at > timedelta(
            minutes=SESSION_TTL_MINUTES
        ):
            expired.append(k)
    for k in expired:
        SESSIONS.pop(k, None)


# ============================================================
# LOOKUP LISTS FOR AREAS + DAMAGE TYPES
# ============================================================

ALLOWED_AREAS = [
    "front bumper upper",
    "front bumper lower",
    "rear bumper upper",
    "rear bumper lower",
    "front left fender",
    "front right fender",
    "rear left fender",
    "rear right fender",
    "front left door",
    "front right door",
    "rear left door",
    "rear right door",
    "left quarter panel",
    "right quarter panel",
    "hood",
    "roof",
    "trunk",
    "tailgate",
    "windshield",
    "rear window",
    "left windows",
    "right windows",
    "left side mirror",
    "right side mirror",
    "left headlight",
    "right headlight",
    "left taillight",
    "right taillight",
    "left front wheel",
    "right front wheel",
    "left rear wheel",
    "right rear wheel",
    "left front tire",
    "right front tire",
    "left rear tire",
    "right rear tire",
]

ALLOWED_DAMAGE_TYPES = [
    "light scratch",
    "deep scratch",
    "paint scuff",
    "paint transfer",
    "small dent",
    "deep dent",
    "dent",
    "dented",
    "heavy dent",
    "heavily dented",
    "crease",
    "panel deformation",
    "bumper deformation",
    "plastic tear",
    "crack",
    "hole",
    "chip",
    "glass chip",
    "glass crack",
    "curb rash",
    "bent wheel",
    "misalignment",
    "deformed",
    "heavily deformed",
]


def clean_areas_from_text(text: str) -> List[str]:
    t = text.lower()
    found = [a for a in ALLOWED_AREAS if a in t]
    out: List[str] = []
    seen = set()
    for a in found:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def clean_damage_types_from_text(text: str) -> List[str]:
    t = text.lower()
    found = [d for d in ALLOWED_DAMAGE_TYPES if d in t]
    out: List[str] = []
    seen = set()
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ============================================================
# TWILIO MEDIA HELPERS
# ============================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")


def download_twilio_media(url: str) -> bytes:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio credentials not set")
    resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    resp.raise_for_status()
    return resp.content


def bytes_to_data_url(data: bytes, ctype: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{ctype};base64,{b64}"


# ============================================================
# VIN DECODER (AI)
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
    vin = vin.strip().upper()
    if len(vin) != 17 or not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        return {
            "vin": vin,
            "year": "unknown",
            "make": "unknown",
            "model": "unknown",
            "body_style": "unknown",
        }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VIN_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": vin}]},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)

    return {
        "vin": vin,
        "year": str(data.get("year", "unknown")),
        "make": str(data.get("make", "unknown")),
        "model": str(data.get("model", "unknown")),
        "body_style": str(data.get("body_style", "unknown")),
    }


# ============================================================
# PRE-SCAN (MULTI-IMAGE FUSION, DRIVER POV L/R)
# ============================================================

PRE_SCAN_SYSTEM_PROMPT = """
You are an automotive DAMAGE PRE-SCAN AI.

You will be given between 1 and 3 photos of a vehicle for a collision center.

Your job:
- Look at ALL photos together and FUSE what you see into ONE combined understanding of the damage.
- Identify ONLY the panels/areas that clearly show visible damage.
- Identify ONLY basic damage types (scratches, dents, cracks, deformation, etc.).
- DO NOT guess damage on panels that are not clearly visible or clearly hit.

LEFT/RIGHT RULE (IMPORTANT):
- Always use the DRIVER'S PERSPECTIVE for left/right (sitting inside the car facing forward).
- NEVER flip orientation based on camera angle.
- If angle is ambiguous, you may say “front-left (likely)” or “front-right (likely)” in NOTES,
  but the AREAS list must still use standard panel names (e.g. "front left fender").

MULTI-IMAGE FUSION:
- Some damage may be more obvious in one photo than another (e.g., hood buckling in one picture, bumper damage in another).
- Combine evidence across ALL photos before deciding which panels are damaged.
- If multiple angles confirm a single impact zone (e.g., front-left hit across bumper, fender, hood),
  reflect that continuity in the AREAS list.

FORMAT TO USE:

ORIENTATION:
- one sentence about the overall camera angles used (e.g. "Photos from front-left and straight-on.")

AREAS:
- front left fender
- front bumper upper
- hood
- left headlight

DAMAGE TYPES:
- deep dent
- panel deformation
- crack
- paint scuff

NOTES:
- short comments about what you see, mention any possible suspension/structural concerns, and any "likely" orientation decisions.
""".strip()


def run_pre_scan(image_data_urls: List[str], shop: Shop) -> Dict[str, Any]:
    if not image_data_urls:
        return {"areas": [], "damage_types": [], "notes": "", "raw": ""}

    content: List[dict] = [
        {
            "type": "text",
            "text": (
                f"Customer photos for {shop.name}. "
                "Follow the PRE-SCAN instructions exactly and fuse observations across ALL photos."
            ),
        }
    ]
    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": PRE_SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    raw = completion.choices[0].message.content or ""

    areas = clean_areas_from_text(raw)
    damage_types = clean_damage_types_from_text(raw)

    notes = ""
    m = re.search(r"NOTES:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        notes = m.group(1).strip()

    return {
        "areas": areas,
        "damage_types": damage_types,
        "notes": notes,
        "raw": raw,
    }


def build_pre_scan_sms(
    shop: Shop, pre: Dict[str, Any], vin_info: Optional[Dict[str, str]] = None
) -> str:
    areas = pre.get("areas", [])
    damage_types = pre.get("damage_types", [])
    notes = pre.get("notes", "") or ""

    lines: List[str] = [f"AI Pre-Scan for {shop.name}", ""]

    if vin_info and vin_info.get("vin"):
        lines.append(
            f"Vehicle (from VIN): {vin_info.get('year', 'unknown')} "
            f"{vin_info.get('make', 'unknown')} {vin_info.get('model', 'unknown')}"
        )
        lines.append("")

    if areas:
        lines.append("From your photo(s), I can clearly see damage on:")
        for a in areas:
            lines.append(f"- {a}")
    else:
        lines.append(
            "I couldn't confidently pick out specific damaged panels yet from these angles."
        )

    if damage_types:
        lines.append("")
        lines.append("Damage types I can see:")
        lines.append("- " + ", ".join(damage_types))

    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.append(notes)

    lines.append("")
    lines.append(
        "If this looks roughly correct, reply 1 and I'll send a full estimate with cost."
    )
    lines.append("If it's off, reply 2 and you can send clearer / wider photos.")
    lines.append("")
    lines.append(
        "Optional: you can also text your 17-character VIN to decode your vehicle details."
    )

    return "\n".join(lines)
    # ============================================================
# ESTIMATE SYSTEM (AI) — severity, line items, pricing logic
# ============================================================

ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, Canada (year 2025).

You receive:
- CONFIRMED damaged areas (already verified)
- CONFIRMED damage types (already verified)
- Optional notes (for example: “hood is heavily dented and deformed on the left side.”)

Your job:
1. Determine SEVERITY: one of ["minor", "moderate", "severe"].
2. Create LINE ITEMS with parts + labour + paint + materials.
   Examples:
      { "panel": "front left fender", "operation": "repair", "hours_body": 2.5, "paint": 1.0 }
      { "panel": "hood", "operation": "replace", "part_cost": 720, "hours_body": 1.0, "paint": 1.2 }
3. DO NOT guess hidden damage (frame, suspension, sensors) unless it is VERY visually obvious.
4. Use conservative but realistic Ontario 2025 estimates.
5. If crack/split/tear in bumper → usually REPLACE.
6. If deep dent or deformation on metal panel → usually REPAIR unless severe.

Respond in strict JSON:
{
  "severity": "minor/moderate/severe",
  "line_items": [
      {
        "panel": "...",
        "operation": "repair/replace",
        "hours_body": number,
        "paint": number,
        "part_cost": number or 0
      }
  ]
}
""".strip()


def run_ai_estimator(areas: List[str], damage_types: List[str], notes: str) -> Dict[str, Any]:
    """Call OpenAI to produce severity + line items."""
    user_payload = {
        "damaged_areas": areas,
        "damage_types": damage_types,
        "notes": notes,
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)

    severity = data.get("severity", "unknown").lower().strip()
    if severity not in ["minor", "moderate", "severe"]:
        severity = "moderate"

    line_items = data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []

    return {
        "severity": severity,
        "line_items": line_items,
    }


# ============================================================
# LEVEL-C PRICING ENGINE (Shop-specific)
# ============================================================

def calculate_costs_with_shop_pricing(
    shop: Shop, ai_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Applies shop-specific labor + materials + part costs + severity floor."""
    if not shop.pricing:
        return {
            "severity": ai_result["severity"],
            "min_cost": 500,
            "max_cost": 2000,
            "line_items": ai_result["line_items"],
        }

    p = shop.pricing
    lr = p.labor_rates
    base = p.base_floor

    severity = ai_result.get("severity", "moderate")

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

        labour_cost = hours_body * lr.body + hours_paint * lr.paint
        materials = hours_paint * p.materials_rate

        if part_cost > 0:
            total_min += labour_cost + materials + part_cost * 0.9
            total_max += labour_cost + materials + part_cost * 1.1
        else:
            total_min += labour_cost + materials
            total_max += labour_cost + materials

    total_min = max(total_min, floor_min)
    total_max = max(total_max, floor_max)

    return {
        "severity": severity,
        "min_cost": int(total_min),
        "max_cost": int(total_max),
        "line_items": ai_result["line_items"],
    }


def run_estimate(shop: Shop, areas: List[str], damage_types: List[str], notes: str) -> Dict[str, Any]:
    """Full estimate pipeline: AI → Level-C pricing."""
    ai_result = run_ai_estimator(areas, damage_types, notes)
    priced = calculate_costs_with_shop_pricing(shop, ai_result)
    return priced


# ============================================================
# ESTIMATE SMS BUILDER
# ============================================================

def build_estimate_sms(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
    vin_info: Optional[Dict[str, str]] = None,
) -> str:
    lines = [f"AI Damage Estimate for {shop.name}", ""]

    if vin_info and vin_info.get("vin"):
        lines.append(
            f"Vehicle (from VIN): {vin_info.get('year')} "
            f"{vin_info.get('make')} {vin_info.get('model')}"
        )
        lines.append("")

    sv = estimate.get("severity", "unknown")
    min_c = estimate.get("min_cost", 0)
    max_c = estimate.get("max_cost", 0)

    if sv == "minor":
        sev_label = "Minor"
        range_text = f"${min_c} – ${max_c}"
    elif sv == "moderate":
        sev_label = "Moderate"
        range_text = f"${min_c} – ${max_c}"
    else:
        sev_label = "Severe"
        range_text = f"${min_c} – ${max_c}"

    lines.append(f"Severity: {sev_label}")
    lines.append(f"Estimated Cost (Ontario 2025): {range_text}")
    lines.append("")

    if areas:
        lines.append("Areas:")
        lines.append("- " + ", ".join(areas))
        lines.append("")

    if damage_types:
        lines.append("Damage Types:")
        lines.append("- " + ", ".join(damage_types))
        lines.append("")

    return "\n".join(lines)
    # ============================================================
# BOOKING UTILITIES & SCHEDULING LOGIC
# ============================================================

APPOINTMENT_DURATION_HOURS = 1
MIN_LEAD_MINUTES = 60  # must be booked at least 1h ahead


def parse_appointment_datetime(text: str) -> Optional[datetime]:
    """Natural-language date/time parser."""
    try:
        dt = date_parser.parse(text, fuzzy=True)
        if dt.tzinfo:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except Exception:
        return None


def get_shop_hours_for_day(shop: Shop, dt: datetime) -> Optional[Tuple[time, time]]:
    """Returns (open, close) for that weekday, or None if closed."""
    day_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    key = day_keys[dt.weekday()]
    hours_cfg = (shop.hours or DEFAULT_HOURS).get(key) or []
    if len(hours_cfg) != 2:
        return None
    try:
        start = datetime.strptime(hours_cfg[0], "%H:%M").time()
        end = datetime.strptime(hours_cfg[1], "%H:%M").time()
        return (start, end)
    except:
        return None


def is_within_shop_hours(shop: Shop, dt: datetime) -> bool:
    """Checks if the given datetime occurs during shop business hours."""
    h = get_shop_hours_for_day(shop, dt)
    if not h:
        return False
    start, end = h
    return start <= dt.time() < end


def is_slot_available(db, shop: Shop, dt: datetime) -> bool:
    """Check if appointment slot is free."""
    start = dt
    end = dt + timedelta(hours=APPOINTMENT_DURATION_HOURS)
    existing = (
        db.query(Booking)
        .filter(
            Booking.shop_id == shop.id,
            Booking.appointment_time >= start,
            Booking.appointment_time < end,
        )
        .all()
    )
    return len(existing) == 0


def format_slot(dt: datetime) -> str:
    """Friendly text for SMS output."""
    return dt.strftime("%a %b %d %I:%M%p").replace("AM", "am").replace("PM", "pm")


def find_next_available_slots(
    db, shop: Shop, start_from: datetime, count: int = 3
) -> List[datetime]:
    """Finds next few open 30-min slots."""
    slots = []
    cur = start_from
    end_limit = start_from + timedelta(days=14)

    while cur < end_limit and len(slots) < count:
        if is_within_shop_hours(shop, cur):
            if cur > datetime.utcnow() + timedelta(minutes=MIN_LEAD_MINUTES):
                if is_slot_available(db, shop, cur):
                    slots.append(cur)
        cur += timedelta(minutes=30)

    return slots


# ============================================================
# BOOKING PARSER UTILITIES (any order)
# ============================================================

def extract_email(text: str) -> Optional[str]:
    """Extract an email anywhere in message."""
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def extract_name_part(body: str, email: Optional[str]) -> str:
    """Take everything before email as the name portion."""
    if not email:
        return ""
    name_chunk = body.split(email)[0]
    name_chunk = (
        name_chunk.replace("Book", "")
        .replace("book", "")
        .replace(",", " ")
        .strip()
    )
    return name_chunk or ""


def booking_like_message(body: str, lower_body: str, email: Optional[str], dt: Optional[datetime]) -> bool:
    """Decide if user intent is booking."""
    if email:
        return True
    if "book" in lower_body:
        return True
    if dt:
        return True
    return False


# ============================================================
# GOOGLE CALENDAR EVENT CREATION SUPPORT (Part 4 uses this)
# ============================================================

def create_calendar_event_for_booking(
    shop: Shop,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    appt_start: datetime,
    session: Dict[str, Any],
) -> Tuple[bool, str, Optional[str]]:
    calendar_id = shop.calendar_id or os.getenv("GOOGLE_CALENDAR_ID")
    if not calendar_id:
        return False, "Calendar ID not configured.", None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return False, "Google Calendar libraries not installed.", None

    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        return False, "Google service account JSON missing.", None

    try:
        info = json.loads(service_account_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=creds)
    except Exception as e:
        return False, f"Failed to init Calendar client: {e}", None

    start_dt = appt_start
    end_dt = appt_start + timedelta(hours=1)

    vin_info = session.get("vin_info") or {}
    estimate = session.get("last_estimate") or {}
    areas = session.get("areas") or []
    damage_types = session.get("damage_types") or []
    photo_links = session.get("photo_links") or []

    desc = []
    desc.append(f"Customer: {customer_name}")
    desc.append(f"Phone: {customer_phone}")
    desc.append(f"Email: {customer_email}")
    desc.append("")

    if photo_links:
        desc.append("Photo links:")
        for url in photo_links:
            desc.append(url)
        desc.append("")

    if vin_info.get("vin"):
        desc.append(
            f"Vehicle VIN: {vin_info.get('vin')} "
            f"({vin_info.get('year')} {vin_info.get('make')} {vin_info.get('model')})"
        )
        desc.append("")

    if areas:
        desc.append("Damaged areas:")
        for a in areas:
            desc.append(f"- {a}")

    if damage_types:
        desc.append("")
        desc.append("Damage types:")
        desc.append(", ".join(damage_types))

    if estimate:
        desc.append("")
        desc.append(
            f"AI Estimate: {estimate.get('severity')} "
            f"${estimate.get('min_cost')}–${estimate.get('max_cost')}"
        )

    event = {
        "summary": f"Repair booking - {customer_name}",
        "description": "\n".join(desc),
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Toronto"},
        "attendees": [{"email": customer_email}],
    }

    try:
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        return True, "Booking added to calendar.", created.get("id")
    except Exception as e:
        return False, f"Calendar error: {e}", None


# ============================================================
# MAIN BOOKING HANDLER
# ============================================================

def handle_booking_request(
    shop: Shop,
    body: str,
    lower_body: str,
    from_number: str,
    session: Dict[str, Any],
) -> Optional[str]:
    """Full booking flow. Returns SMS response or None."""
    
    # Detect email anywhere
    email = extract_email(body)

    # Detect datetime anywhere
    appt_dt = parse_appointment_datetime(body)

    # Detect “only time” → ask for date
    only_time = re.fullmatch(r"\s*\d{1,2}\s*(am|pm)\s*", lower_body)
    if only_time:
        return (
            "Please include a DATE along with the time.\n"
            "Example: John Doe, email@example.com, Nov 29 3pm"
        )

    # Not a booking-like message
    if not booking_like_message(body, lower_body, email, appt_dt):
        return None

    # Missing datetime
    if appt_dt is None:
        return (
            "I need both DATE and TIME to book an appointment.\n"
            "Example: John Doe, email@example.com, Nov 29 3pm"
        )

    # Missing email (required)
    if not email:
        return (
            "I can reserve that time, but I need your EMAIL to complete booking.\n"
            "Please send name, email, and date/time together.\n"
            "Example: John Doe, email@example.com, Nov 29 3pm"
        )

    # Extract name (everything before email)
    full_name = extract_name_part(body, email)
    if not full_name:
        full_name = "Customer"

    # Must be in the future
    now = datetime.utcnow()
    if appt_dt <= now + timedelta(minutes=MIN_LEAD_MINUTES):
        with db_session() as db:
            suggestions = find_next_available_slots(
                db, shop, now + timedelta(minutes=MIN_LEAD_MINUTES)
            )
        if suggestions:
            msg = ["That time is too soon.", ""]
            msg.append("Next available times:")
            for s in suggestions:
                msg.append(f"- {format_slot(s)}")
            msg.append("")
            msg.append("Reply with your name, email, and chosen time.")
            return "\n".join(msg)
        else:
            return "That time is too soon. Please pick another future time."

    # Hours + availability validation
    with db_session() as db:

        # Closed that day
        hours = get_shop_hours_for_day(shop, appt_dt)
        if not hours:
            suggestions = find_next_available_slots(db, shop, appt_dt)
            msg = ["The shop is CLOSED at that time."]
            if suggestions:
                msg.append("")
                msg.append("Next available times:")
                for s in suggestions:
                    msg.append(f"- {format_slot(s)}")
                msg.append("")
                msg.append("Reply with your name, email, and chosen time.")
            return "\n".join(msg)

        # Outside hours
        if not is_within_shop_hours(shop, appt_dt):
            suggestions = find_next_available_slots(db, shop, appt_dt)
            msg = ["That time is OUTSIDE shop hours."]
            if suggestions:
                msg.append("")
                msg.append("Open times:")
                for s in suggestions:
                    msg.append(f"- {format_slot(s)}")
                msg.append("")
                msg.append("Reply with your name, email, and chosen time.")
            return "\n".join(msg)

        # Double-booking
        if not is_slot_available(db, shop, appt_dt):
            suggestions = find_next_available_slots(db, shop, appt_dt)
            msg = ["That time is already BOOKED."]
            if suggestions:
                msg.append("")
                msg.append("Other times available:")
                for s in suggestions:
                    msg.append(f"- {format_slot(s)}")
                msg.append("")
                msg.append("Reply with your name, email, and chosen time.")
            return "\n".join(msg)

        # Slot is valid — save in DB + Calendar
        customer = get_or_create_customer(
            db, phone=from_number, full_name=full_name, email=email
        )

        success, calendar_msg, event_id = create_calendar_event_for_booking(
            shop,
            customer_name=full_name,
            customer_email=email,
            customer_phone=from_number,
            appt_start=appt_dt,
            session=session,
        )

        save_booking_record(
            db,
            customer=customer,
            shop_id=shop.id,
            appointment_time=appt_dt,
            calendar_event_id=event_id,
        )

    # Final confirmation message
    if success:
        return f"Your appointment is booked!\nTime: {format_slot(appt_dt)}\n{calendar_msg}"
    else:
        return (
            f"Your booking was saved but calendar failed:\n{calendar_msg}\n"
            f"Requested time: {format_slot(appt_dt)}"
        )
        # ============================================================
# TWILIO WEBHOOK: PHOTOS → PRE-SCAN → ESTIMATE → BOOKING
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token) if token else None

    reply = MessagingResponse()

    if not shop:
        reply.message(
            "This number is not configured for a shop. "
            "Please contact the collision centre directly."
        )
        return Response(content=str(reply), media_type="application/xml")

    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        lower_body = body.lower()
        from_number = form.get("From", "")
        num_media_str = form.get("NumMedia", "0") or "0"

        try:
            num_media = int(num_media_str)
        except ValueError:
            num_media = 0

        # Load / init session
        cleanup_sessions()
        key = session_key(shop, from_number)
        session = SESSIONS.get(key) or {}

        # ----------------------------------------------------
        # A) MMS photos → AI PRE-SCAN
        # ----------------------------------------------------
        if num_media > 0:
            image_data_urls: List[str] = []
            media_links: List[str] = []

            for i in range(num_media):
                media_url = form.get(f"MediaUrl{i}")
                ctype = form.get(f"MediaContentType{i}") or "image/jpeg"
                if not media_url:
                    continue

                media_links.append(media_url)

                try:
                    raw = download_twilio_media(media_url)
                except Exception:
                    reply.message(
                        "Sorry — I couldn't download the photos. "
                        "Please resend 1–3 clear photos of the damaged area."
                    )
                    return Response(content=str(reply), media_type="application/xml")

                data_url = bytes_to_data_url(raw, ctype)
                image_data_urls.append(data_url)

            if not image_data_urls:
                reply.message(
                    "Sorry — I couldn't process the photos. "
                    "Please resend 1–3 clear photos of the damaged area."
                )
                return Response(content=str(reply), media_type="application/xml")

            pre_scan = run_pre_scan(image_data_urls, shop)

            session.update(
                {
                    "areas": pre_scan.get("areas", []),
                    "damage_types": pre_scan.get("damage_types", []),
                    "notes": pre_scan.get("notes", ""),
                    "raw_pre_scan": pre_scan.get("raw", ""),
                    "photo_links": media_links,
                    "created_at": datetime.utcnow(),
                }
            )
            SESSIONS[key] = session

            vin_info = session.get("vin_info")
            text = build_pre_scan_sms(shop, pre_scan, vin_info)
            reply.message(text)
            return Response(content=str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # B) Reply 1 → confirm pre-scan → run ESTIMATE
        # ----------------------------------------------------
        if lower_body in {"1", "yes", "y"}:
            if not session.get("areas") and not session.get("damage_types"):
                reply.message(
                    "I don't see a recent photo for this number. "
                    "Please send 1–3 clear photos of the damaged area to start a new estimate."
                )
                return Response(content=str(reply), media_type="application/xml")

            vin_info = session.get("vin_info")

            try:
                estimate = run_estimate(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    session.get("notes", ""),
                )
                session["last_estimate"] = estimate
                SESSIONS[key] = session

                # Save estimate + vehicle info in DB
                try:
                    with db_session() as db:
                        customer = get_or_create_customer(db, phone=from_number)
                        if vin_info:
                            upsert_vehicle_from_vin(db, customer, vin_info)
                        save_estimate_record(
                            db,
                            customer=customer,
                            shop_id=shop.id,
                            areas=session.get("areas", []),
                            damage_types=session.get("damage_types", []),
                            estimate=estimate,
                            notes=session.get("notes", ""),
                        )
                except Exception as db_err:
                    print("DB error while saving estimate:", repr(db_err))
                    traceback.print_exc()

                text = build_estimate_sms(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    estimate,
                    vin_info=vin_info,
                )
                reply.message(text)
                return Response(content=str(reply), media_type="application/xml")

            except Exception as e:
                print("ERROR in estimate generation:", repr(e))
                traceback.print_exc()
                reply.message(
                    "Sorry — I couldn't generate your estimate just now. "
                    "Please resend the photos to start again."
                )
                return Response(content=str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # C) Reply 2 → pre-scan incorrect → reset session
        # ----------------------------------------------------
        if lower_body in {"2", "no", "n"}:
            SESSIONS.pop(key, None)
            reply.message(
                "No problem. Please send 1–3 clearer photos of the damaged area "
                "(a wide shot plus a couple close-ups) and I’ll rescan it."
            )
            return Response(content=str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # D) VIN handling — 17-character VIN text
        # ----------------------------------------------------
        stripped = body.replace(" ", "").upper()
        if len(stripped) == 17 and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", stripped):
            vin_info = decode_vin_with_ai(stripped)
            session["vin_info"] = vin_info
            if "created_at" not in session:
                session["created_at"] = datetime.utcnow()
            SESSIONS[key] = session

            # Save/update vehicle in DB
            with db_session() as db:
                customer = get_or_create_customer(db, phone=from_number)
                upsert_vehicle_from_vin(db, customer, vin_info)

            msg_lines = [
                "VIN decoded:",
                f"VIN: {vin_info.get('vin')}",
                f"Year: {vin_info.get('year')}",
                f"Make: {vin_info.get('make')}",
                f"Model: {vin_info.get('model')}",
                f"Body style: {vin_info.get('body_style')}",
            ]
            reply.message("\n".join(msg_lines))
            return Response(content=str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # E) Booking — flexible ANY-ORDER booking handler
        # ----------------------------------------------------
        booking_response = handle_booking_request(
            shop=shop,
            body=body,
            lower_body=lower_body,
            from_number=from_number,
            session=session,
        )
        if booking_response is not None:
            reply.message(booking_response)
            SESSIONS[key] = session
            return Response(content=str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # F) Default instructions
        # ----------------------------------------------------
        instructions = (
            f"Hi from {shop.name}!\n\n"
            "To get an AI-powered damage estimate:\n"
            "1) Send 1–3 clear photos of the damaged area.\n"
            "2) I’ll analyze them and send an AI Pre-Scan.\n"
            "3) Reply 1 if it looks right, or 2 if it’s off.\n"
            "4) Then I'll send your full Ontario 2025 cost estimate.\n\n"
            "Optional:\n"
            "- Text your 17-character VIN to decode your vehicle details.\n"
            "- After your estimate, book a repair by replying with your name, email, and date/time (any order), for example:\n"
            "  John Doe, email@example.com, Nov 29 3pm"
        )
        reply.message(instructions)
        return Response(content=str(reply), media_type="application/xml")

    except Exception as e:
        print("UNEXPECTED ERROR in sms-webhook:", repr(e))
        traceback.print_exc()
        reply.message(
            "Unexpected error on our side. Please try again in a moment or resend your photos."
        )
        return Response(content=str(reply), media_type="application/xml")


# ============================================================
# ADMIN + HEALTHCHECK
# ============================================================

@app.get("/admin/shops")
async def list_shops():
    base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    result = []
    for shop in SHOPS_BY_TOKEN.values():
        webhook = (
            f"{base_url}/sms-webhook?token={shop.webhook_token}"
            if base_url
            else f"/sms-webhook?token={shop.webhook_token}"
        )
        result.append(
            {
                "id": shop.id,
                "name": shop.name,
                "webhook_token": shop.webhook_token,
                "calendar_id": shop.calendar_id,
                "pricing": shop.pricing.dict() if shop.pricing else None,
                "hours": shop.hours or DEFAULT_HOURS,
                "twilio_webhook_example": webhook,
            }
        )
    return result


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI damage estimator with VIN, fusion pre-scan, Level-C pricing, booking, and calendar integration is running.",
        }
