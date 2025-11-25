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

# Natural-language datetime parsing
from dateutil import parser as date_parser

# ===========================
# SQLAlchemy (PostgreSQL ONLY)
# ===========================
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
# FastAPI + OpenAI setup
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Multi-shop config (tokenized routing via SHOPS_JSON)
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


# Default hours for all shops (your schedule)
DEFAULT_HOURS: Dict[str, List[str]] = {
    "mon": ["09:00", "17:00"],
    "tue": ["09:00", "17:00"],
    "wed": ["09:00", "19:00"],
    "thu": ["09:00", "19:00"],
    "fri": ["09:00", "19:00"],
    "sat": ["09:00", "17:00"],
    "sun": [],  # CLOSED
}


class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str  # used as ?token=... in Twilio URL
    calendar_id: Optional[str] = None  # optional Google Calendar ID
    pricing: Optional[ShopPricing] = None  # Level-C pricing
    hours: Optional[Dict[str, List[str]]] = None  # business hours per day


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
# In-memory session store (per shop+phone)
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
        if isinstance(created_at, datetime):
            if now - created_at > timedelta(minutes=SESSION_TTL_MINUTES):
                expired.append(k)
    for k in expired:
        SESSIONS.pop(k, None)

# ============================================================
# Allowed areas + damage vocab
# ============================================================

ALLOWED_AREAS = [
    "front bumper upper", "front bumper lower",
    "rear bumper upper", "rear bumper lower",
    "front left fender", "front right fender",
    "rear left fender", "rear right fender",
    "front left door", "front right door",
    "rear left door", "rear right door",
    "left quarter panel", "right quarter panel",
    "hood", "roof", "trunk", "tailgate",
    "windshield", "rear window",
    "left windows", "right windows",
    "left side mirror", "right side mirror",
    "left headlight", "right headlight",
    "left taillight", "right taillight",
    "left front wheel", "right front wheel",
    "left rear wheel", "right rear wheel",
    "left front tire", "right front tire",
    "left rear tire", "right rear tire",
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
# Media download helpers (Twilio)
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
# VIN decoding via OpenAI (no external VIN API)
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
# PRE-SCAN (multi-image fusion + left/right fix)
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
# ESTIMATE + line-item breakdown (Audatex-style)
# ============================================================

ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, Canada (year 2025).

You receive:
- CONFIRMED damaged areas (from a photo-based pre-scan).
- CONFIRMED basic damage types.
- Optional notes (for example: “hood is heavily dented and deformed on the left side.”).

Your tasks:
1) Classify severity: "minor", "moderate", "severe", or "unknown".
2) Provide a realistic repair cost range in CAD (min_cost, max_cost) for labour + materials.
3) Write a short, customer-friendly explanation (2–4 sentences).
4) Produce a simple Audatex-style line-item breakdown (operations, hours, parts).
5) Include a brief disclaimer that this is a visual preliminary estimate only.

Severity guidance:
- Crushed, heavily dented, caved-in, or clearly deformed panels on structural areas (trunk, hood, bumpers, quarter panels, roof)
  are usually "moderate" or "severe", not "minor".
- Small cosmetic scratches or scuffs on a single panel with no deformation are usually "minor".
- Multiple damaged panels or combined deformation + cracks typically mean "moderate" or "severe".

STRICT RULES:
- NEVER add new panels or areas beyond those given.
- If confirmed areas list is empty, set severity to "unknown" and keep cost very low or 0–200.

Line items:
- Use generic operations like "R&R", "Repair", "Refinish", "Blend", "R&I".
- Use rough labour hours (e.g. 0.5, 1.2, 2.0).
- Use placeholder part descriptions like "Fender OEM part" or "Bumper cover (aftermarket)" without prices.
- Keep the total number of line items small (max 8–10) so it fits in an SMS.

OUTPUT JSON ONLY IN THIS EXACT SCHEMA:

{
  "severity": "minor | moderate | severe | unknown",
  "min_cost": 0,
  "max_cost": 0,
  "summary": "2–4 sentence explanation.",
  "disclaimer": "Short note reminding it's a visual estimate only.",
  "line_items": [
    {
      "panel": "front left fender",
      "operation": "R&R",
      "hours": 2.5,
      "notes": "Remove and replace damaged fender."
    }
  ]
}
""".strip()


def run_estimate(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    payload = {
        "shop_name": shop.name,
        "region": "Ontario",
        "year": 2025,
        "confirmed_areas": areas,
        "damage_types": damage_types,
        "pre_scan_notes": notes or "",
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)

    severity = (data.get("severity") or "unknown").lower()
    if severity not in {"minor", "moderate", "severe", "unknown"}:
        severity = "unknown"

    def _f(v, default=0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    estimate = {
        "severity": severity,
        "min_cost": _f(data.get("min_cost", 0)),
        "max_cost": _f(data.get("max_cost", 0)),
        "summary": str(data.get("summary", "")).strip(),
        "disclaimer": str(data.get("disclaimer", "")).strip(),
        "line_items": data.get("line_items", []) or [],
    }

    return sanity_adjust_estimate(shop, estimate, areas, damage_types, notes)


# -------- Level-C hours + pricing helpers --------

def _estimate_hours_for_panel(
    panel: str, damage_types: List[str], severity: str
) -> Tuple[float, float, float]:
    """
    Rough heuristic for body, paint, frame hours per panel.
    Returns (body_hours, paint_hours, frame_hours).
    """
    p = panel.lower()
    dmg_text = " ".join(damage_types).lower()

    body_hours = 0.0
    paint_hours = 0.0
    frame_hours = 0.0

    structural_panels = [
        "hood",
        "trunk",
        "roof",
        "left quarter panel",
        "right quarter panel",
        "front bumper upper",
        "rear bumper upper",
        "front bumper lower",
        "rear bumper lower",
    ]

    scratch_like = any(
        k in dmg_text
        for k in ["scratch", "scuff", "paint transfer", "chip", "curb rash"]
    )
    dent_like = any(
        k in dmg_text for k in ["dent", "dented", "crease", "deformation", "deformed"]
    )
    crack_like = any(k in dmg_text for k in ["crack", "hole", "tear"])
    heavy_like = any(
        k in dmg_text
        for k in [
            "deep dent",
            "heavy dent",
            "heavily dented",
            "heavily deformed",
            "crushed",
            "caved in",
            "buckled",
            "pushed in",
            "folded",
        ]
    )

    # Wheels / tires: lighter structure work
    if "wheel" in p or "tire" in p or "rim" in p:
        if scratch_like:
            body_hours += 0.4
            paint_hours += 0.7
        elif dent_like or crack_like:
            body_hours += 1.0
            paint_hours += 1.0
        return body_hours, paint_hours, frame_hours

    # Base hours by severity
    if scratch_like and not dent_like and not crack_like:
        # Cosmetic only
        if severity == "minor":
            body_hours += 0.3
            paint_hours += 0.7
        elif severity == "moderate":
            body_hours += 0.5
            paint_hours += 1.2
        else:  # severe or unknown
            body_hours += 0.7
            paint_hours += 1.5
    elif dent_like or crack_like:
        # Real repair
        if severity == "minor":
            body_hours += 1.0
            paint_hours += 1.5
        elif severity == "moderate":
            body_hours += 2.0
            paint_hours += 2.0
        else:  # severe
            body_hours += 3.0
            paint_hours += 2.5

    if crack_like:
        # Extra work for cracks / holes, especially bumpers
        body_hours += 0.5
        paint_hours += 0.5

    if p in structural_panels and heavy_like:
        # Structural + heavy damage → some frame time
        frame_hours += 1.0
        if severity == "severe":
            frame_hours += 1.0

    return body_hours, paint_hours, frame_hours


def _compute_pricing_from_hours(
    pricing: ShopPricing,
    areas: List[str],
    damage_types: List[str],
    severity: str,
    notes: str = "",
) -> Tuple[float, Dict[str, float]]:
    """
    Level-C pricing: estimate body/paint/frame hours from panels + damage types,
    then multiply by labour rates + materials.
    Returns (total_cost, hours_breakdown).
    """
    if not pricing or not pricing.labor_rates:
        return 0.0, {"body": 0.0, "paint": 0.0, "frame": 0.0}

    body_total = 0.0
    paint_total = 0.0
    frame_total = 0.0

    lowered_areas = [a.lower() for a in areas]
    dmg = [d.lower() for d in damage_types]

    # If nothing is identified, nothing to compute
    if not lowered_areas:
        return 0.0, {"body": 0.0, "paint": 0.0, "frame": 0.0}

    for panel in lowered_areas:
        b, p, f = _estimate_hours_for_panel(panel, dmg, severity)
        body_total += b
        paint_total += p
        frame_total += f

    # If we have multiple panels, add a bit of overlap/complexity
    if len(lowered_areas) >= 2:
        body_total *= 1.1
        paint_total *= 1.05

    # Pull rates
    body_rate = pricing.labor_rates.body
    paint_rate = pricing.labor_rates.paint
    frame_rate = pricing.labor_rates.frame or pricing.labor_rates.body

    labour_cost = (
        body_total * body_rate
        + paint_total * paint_rate
        + frame_total * frame_rate
    )

    # Materials – one bucket, scaled slightly by severity & #panels
    materials = pricing.materials_rate
    if len(lowered_areas) >= 3:
        materials *= 1.5
    if severity == "severe":
        materials *= 1.3
    elif severity == "moderate":
        materials *= 1.1

    total_cost = labour_cost + materials

    # Small safety margin
    total_cost *= 1.05

    hours_info = {
        "body": round(body_total, 2),
        "paint": round(paint_total, 2),
        "frame": round(frame_total, 2),
    }
    return total_cost, hours_info


def sanity_adjust_estimate(
    shop: Shop,
    estimate: Dict[str, Any],
    areas: List[str],
    damage_types: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    severity = estimate.get("severity", "unknown").lower()

    def _f(v, default=0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    min_cost = _f(estimate.get("min_cost", 0))
    max_cost = _f(estimate.get("max_cost", 0))

    lowered_areas = [a.lower() for a in areas]
    lowered_types = [d.lower() for d in damage_types]
    notes_text = (notes or "").lower()

    wheel_like = [a for a in lowered_areas if "wheel" in a or "rim" in a]
    tire_like = [a for a in lowered_areas if "tire" in a]
    non_wheel_panels = [a for a in lowered_areas if a not in wheel_like + tire_like]

    # Basic severity ranking
    severity_rank = {"unknown": 0, "minor": 1, "moderate": 2, "severe": 3}
    rank_to_label = {v: k for k, v in severity_rank.items()}
    current_rank = severity_rank.get(severity, 0)

    heavy_terms = [
        "heavily dented",
        "heavy dent",
        "deep dent",
        "panel deformation",
        "bumper deformation",
        "deformed",
        "heavily deformed",
        "crushed",
        "caved in",
        "buckled",
        "pushed in",
        "folded",
    ]

    types_text = " ".join(lowered_types)
    has_heavy = any(term in types_text or term in notes_text for term in heavy_terms)

    structural_panels = [
        "hood",
        "trunk",
        "roof",
        "left quarter panel",
        "right quarter panel",
        "front bumper upper",
        "front bumper lower",
        "rear bumper upper",
        "rear bumper lower",
    ]
    has_structural = any(p in lowered_areas for p in structural_panels)

    # Adjust severity rank based on heavy + structural
    if has_heavy and has_structural:
        current_rank = max(current_rank, severity_rank["severe"])
    elif has_heavy:
        current_rank = max(current_rank, severity_rank["moderate"])

    # Multi-panel heuristic
    if len(non_wheel_panels) >= 3:
        current_rank = max(current_rank, severity_rank["moderate"])

    severity = rank_to_label.get(current_rank, severity)
    estimate["severity"] = severity

    pricing = shop.pricing

    # If we have Level-C pricing, use it
    if pricing and pricing.labor_rates:
        total_cost, hours_info = _compute_pricing_from_hours(
            pricing, areas, damage_types, severity, notes
        )

        if total_cost > 0:
            # Convert AI suggestion into range around the computed cost
            base = total_cost
            min_cost = base * 0.9
            max_cost = base * 1.2

        # Enforce base floors by severity (shop-specific)
        floor = pricing.base_floor
        if severity == "minor":
            min_cost = max(min_cost, floor.minor_min)
            max_cost = max(max_cost, floor.minor_min)
            max_cost = max(min_cost, min(max_cost, floor.minor_max))
        elif severity == "moderate":
            min_cost = max(min_cost, floor.moderate_min)
            max_cost = max(max_cost, floor.moderate_min)
            max_cost = max(min_cost, min(max_cost, floor.moderate_max))
        elif severity == "severe":
            min_cost = max(min_cost, floor.severe_min)
            max_cost = max(max_cost, floor.severe_min)
            max_cost = max(min_cost, min(max_cost, floor.severe_max))
        else:
            # Unknown severity – keep range but at least non-negative
            min_cost = max(min_cost, 0)
            max_cost = max(max_cost, min_cost + 100)

        if max_cost < min_cost:
            max_cost = min_cost

        # Round to nearest $10
        min_cost = int(round(min_cost / 10.0)) * 10
        max_cost = int(round(max_cost / 10.0)) * 10

        estimate["min_cost"] = min_cost
        estimate["max_cost"] = max_cost
        # Optionally could attach hours_info into line_items or hidden field
        return estimate

    # ---------- Fallback legacy pricing if no Level-C pricing configured ----------

    # Wheel-only jobs
    if non_wheel_panels == [] and (wheel_like or tire_like):
        serious = any(
            key in " ".join(lowered_types)
            for key in ["bent wheel", "crack", "deep dent", "hole", "puncture"]
        )
        if serious:
            min_cost = max(min_cost, 250)
            max_cost = max(max_cost, 700)
            severity = "moderate"
        else:
            min_cost = max(min_cost, 120)
            max_cost = max(max_cost, 450)
            severity = "minor"

    # If we still have nothing, fallback by severity only
    if severity == "minor":
        if min_cost <= 0:
            min_cost = 150
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 600
        max_cost = min(max_cost, 2000)
    elif severity == "moderate":
        if min_cost <= 0:
            min_cost = 600
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 2500
    elif severity == "severe":
        if min_cost <= 0:
            min_cost = 2000
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 6000
    else:
        if min_cost <= 0 and max_cost <= 0:
            min_cost, max_cost = 0, 200

    if max_cost < min_cost:
        max_cost = min_cost

    # Round to nearest $10
    min_cost = int(round(min_cost / 10.0)) * 10
    max_cost = int(round(max_cost / 10.0)) * 10

    estimate["severity"] = severity
    estimate["min_cost"] = min_cost
    estimate["max_cost"] = max_cost
    return estimate

# ============================================================
# AI explanation text
# ============================================================

EXPLANATION_SYSTEM_PROMPT = """
You are a service advisor at an auto body shop.

You receive:
- A list of damaged areas.
- A list of damage types.
- A short summary and severity from a previous AI estimate.

Write a short, very clear explanation (2–4 sentences) that:
- Explains WHAT is damaged, panel by panel.
- Explains in simple terms why the severity is minor/moderate/severe.
- Does NOT talk about pricing or dollars.
- Sounds friendly and professional.

Return plain text only (no JSON, no bullet points).
""".strip()


def build_explanation_text(
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
) -> str:
    if not areas and not damage_types:
        return ""

    payload = {
        "areas": areas,
        "damage_types": damage_types,
        "severity": estimate.get("severity", "unknown"),
        "summary": estimate.get("summary", ""),
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    )
    text = completion.choices[0].message.content or ""
    return text.strip()

# ============================================================
# Build SMS with line-items + explanation
# ============================================================

def build_estimate_sms(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
    vin_info: Optional[Dict[str, str]] = None,
) -> str:
    severity = estimate.get("severity", "unknown").capitalize()
    min_cost = int(estimate.get("min_cost", 0) or 0)
    max_cost = int(estimate.get("max_cost", 0) or 0)
    summary = estimate.get("summary") or ""
    disclaimer = estimate.get("disclaimer") or (
        "This is a visual pre-estimate only. Final pricing may change after in-person inspection."
    )
    line_items = estimate.get("line_items") or []

    explanation = build_explanation_text(areas, damage_types, estimate)

    estimate_id = str(uuid.uuid4())
    areas_str = ", ".join(areas) if areas else "not clearly identified yet"
    dmg_str = ", ".join(damage_types) if damage_types else "not clearly classified yet"

    lines: List[str] = [f"AI Damage Estimate for {shop.name}", ""]

    if vin_info and vin_info.get("vin"):
        lines.append(
            f"Vehicle: {vin_info.get('year', 'unknown')} "
            f"{vin_info.get('make', 'unknown')} {vin_info.get('model', 'unknown')}"
        )
        lines.append("")

    lines.extend(
        [
            f"Severity: {severity}",
            f"Estimated Cost (Ontario 2025): ${min_cost:,} – ${max_cost:,}",
            f"Areas: {areas_str}",
            f"Damage Types: {dmg_str}",
        ]
    )

    if summary:
        lines.append("")
        lines.append(summary)

    if line_items:
        lines.append("")
        lines.append("Line-item breakdown:")
        for item in line_items[:6]:
            panel = item.get("panel", "panel")
            op = item.get("operation", "operation")
            hrs = item.get("hours", 0)
            lines.append(f"- {panel}: {op} (~{hrs} hrs)")

    if explanation:
        lines.append("")
        lines.append("Damage overview:")
        lines.append(explanation)

    lines.append("")
    lines.append(f"Estimate ID (internal): {estimate_id}")
    lines.append("")
    lines.append(disclaimer)

    lines.append("")
    lines.append("To book a repair appointment, reply:")
    lines.append("Book Full Name, email@example.com, Nov 28 9am")

    return "\n".join(lines)

# ============================================================
# Google Calendar integration (optional)
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
        return False, "Google service account JSON not configured.", None

    try:
        info = json.loads(service_account_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=creds)
    except Exception as e:
        return False, f"Failed to init Calendar client: {e}", None

    # appt_start is naive local (America/Toronto)
    start_dt = appt_start
    end_dt = appt_start + timedelta(hours=1)

    vin_info = session.get("vin_info") or {}
    estimate = session.get("last_estimate") or {}
    areas = session.get("areas") or []
    damage_types = session.get("damage_types") or []
    photo_links = session.get("photo_links") or []

    description_lines = [
        f"Customer: {customer_name}",
        f"Phone: {customer_phone}",
        f"Email: {customer_email}",
        "",
    ]

    if photo_links:
        description_lines.append("Photo links (Twilio):")
        for url in photo_links:
            description_lines.append(url)
        description_lines.append("")

    if vin_info.get("vin"):
        description_lines.append(
            f"Vehicle VIN: {vin_info.get('vin')} "
            f"({vin_info.get('year', 'unknown')} {vin_info.get('make', 'unknown')} {vin_info.get('model', 'unknown')})"
        )
        description_lines.append("")
    if areas:
        description_lines.append("Damaged areas:")
        for a in areas:
            description_lines.append(f"- {a}")
    if damage_types:
        description_lines.append("")
        description_lines.append("Damage types:")
        description_lines.append(", ".join(damage_types))
    if estimate:
        description_lines.append("")
        description_lines.append(
            f"AI Estimated Severity: {estimate.get('severity', 'unknown')} "
            f"Cost: ${estimate.get('min_cost', 0)}–${estimate.get('max_cost', 0)}"
        )

    event = {
        "summary": f"Repair booking - {customer_name}",
        "description": "\n".join(description_lines),
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Toronto"},
        "attendees": [{"email": customer_email}],
    }

    try:
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        event_id = created.get("id")
        return True, "Booking added to calendar.", event_id
    except Exception as e:
        return False, f"Failed to create calendar event: {e}", None

# ============================================================
# Booking helpers: hours + availability + suggestions
# ============================================================

APPOINTMENT_DURATION_HOURS = 1
MIN_LEAD_MINUTES = 60  # at least 1 hour in advance


def parse_appointment_datetime(text: str) -> datetime:
    """
    Parse natural language date/time like 'Nov 28 9am', '2025-11-28 09:00',
    'tomorrow at 3pm', etc. Returns naive datetime (local).
    """
    dt = date_parser.parse(text, fuzzy=True)
    # Treat as local naive (America/Toronto) without tzinfo
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def get_shop_hours_for_day(shop: Shop, dt: datetime) -> Optional[Tuple[time, time]]:
    day_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    key = day_keys[dt.weekday()]
    hours_cfg = (shop.hours or DEFAULT_HOURS).get(key) or []
    if len(hours_cfg) != 2:
        return None
    start_str, end_str = hours_cfg
    try:
        start_h = datetime.strptime(start_str, "%H:%M").time()
        end_h = datetime.strptime(end_str, "%H:%M").time()
        return start_h, end_h
    except Exception:
        return None


def is_within_shop_hours(shop: Shop, dt: datetime) -> bool:
    h = get_shop_hours_for_day(shop, dt)
    if not h:
        return False
    start_h, end_h = h
    t = dt.time()
    return start_h <= t < end_h


def is_slot_available(db, shop: Shop, dt: datetime) -> bool:
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
    return dt.strftime("%a %b %d %I:%M%p").replace("AM", "am").replace("PM", "pm")


def find_next_available_slots(
    db, shop: Shop, start_from: datetime, count: int = 3
) -> List[datetime]:
    slots: List[datetime] = []
    cur = start_from
    end_limit = start_from + timedelta(days=14)

    while cur < end_limit and len(slots) < count:
        if is_within_shop_hours(shop, cur):
            if cur > datetime.utcnow() + timedelta(minutes=MIN_LEAD_MINUTES):
                if is_slot_available(db, shop, cur):
                    slots.append(cur)
        cur += timedelta(minutes=30)  # scan every 30 minutes

    return slots

# ============================================================
# Twilio webhook (MMS + VIN + estimate + booking + DB)
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token) if token else None

    reply = MessagingResponse()

    if not shop:
        reply.message(
            "This number is not configured correctly. Please contact the body shop directly."
        )
        return Response(content=str(reply), media_type="application/xml")

    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From", "")
        num_media_str = form.get("NumMedia", "0") or "0"

        try:
            num_media = int(num_media_str)
        except ValueError:
            num_media = 0

        cleanup_sessions()
        key = session_key(shop, from_number)
        session = SESSIONS.get(key) or {}
        lower_body = body.lower()

        # A) Incoming MMS -> PRE-SCAN
        if num_media > 0:
            image_data_urls: List[str] = []
            media_links: List[str] = []  # store original Twilio media URLs

            for i in range(num_media):
                media_url = form.get(f"MediaUrl{i}")
                content_type = form.get(f"MediaContentType{i}") or "image/jpeg"
                if not media_url:
                    continue

                media_links.append(media_url)  # keep link for calendar

                try:
                    raw = download_twilio_media(media_url)
                except Exception:
                    reply.message(
                        "Sorry — I couldn't download the photos from your carrier. "
                        "Please resend 1–3 clear photos of the damaged area."
                    )
                    return Response(content=str(reply), media_type="application/xml")

                data_url = bytes_to_data_url(raw, content_type)
                image_data_urls.append(data_url)

            if not image_data_urls:
                reply.message(
                    "Sorry — I had trouble reading the photos. "
                    "Please resend 1–3 clear photos of the damaged area."
                )
            else:
                try:
                    pre_scan = run_pre_scan(image_data_urls, shop)
                except Exception:
                    reply.message(
                        "Sorry — I couldn't process the photos. "
                        "Please try sending them again."
                    )
                    return Response(content=str(reply), media_type="application/xml")

                session.update(
                    {
                        "areas": pre_scan.get("areas", []),
                        "damage_types": pre_scan.get("damage_types", []),
                        "notes": pre_scan.get("notes", ""),
                        "raw_pre_scan": pre_scan.get("raw", ""),
                        "photo_links": media_links,  # for calendar
                        "created_at": datetime.utcnow(),
                    }
                )
                SESSIONS[key] = session

                vin_info = session.get("vin_info")
                reply.message(build_pre_scan_sms(shop, pre_scan, vin_info))

            return Response(content=str(reply), media_type="application/xml")

        # B) Reply 1 -> confirm pre-scan, send estimate (DB failures logged but don't break SMS)
        if lower_body in {"1", "yes", "y"}:
            if not session.get("areas") and not session.get("damage_types"):
                reply.message(
                    "I don't see a recent photo for this number. "
                    "Please send 1–3 clear photos of the damaged area to start a new estimate."
                )
                return Response(content=str(reply), media_type="application/xml")

            vin_info = session.get("vin_info")

            try:
                # 1) Generate estimate first (no DB yet)
                estimate = run_estimate(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    session.get("notes", ""),
                )
                session["last_estimate"] = estimate
                SESSIONS[key] = session

                # 2) Try to save to DB, but don't break SMS if DB fails
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

                # 3) Build the SMS to send to the customer
                text = build_estimate_sms(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    estimate,
                    vin_info=vin_info,
                )

            except Exception as e:
                print("ERROR in estimate generation:", repr(e))
                traceback.print_exc()
                text = (
                    "Sorry — I couldn’t generate the estimate just now. "
                    "Please resend the photos to start again."
                )

            reply.message(text)
            return Response(content=str(reply), media_type="application/xml")

        # C) Reply 2 -> pre-scan wrong
        if lower_body in {"2", "no", "n"}:
            SESSIONS.pop(key, None)
            reply.message(
                "No problem. Please send 1–3 clearer photos of the damaged area "
                "(include a wide shot plus a couple close-ups) and I'll rescan it."
            )
            return Response(content=str(reply), media_type="application/xml")

        # D) VIN message
        stripped = body.replace(" ", "").upper()
        if len(stripped) == 17 and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", stripped):
            vin_info = decode_vin_with_ai(stripped)
            session["vin_info"] = vin_info
            if "created_at" not in session:
                session["created_at"] = datetime.utcnow()
            SESSIONS[key] = session

            with db_session() as db:
                customer = get_or_create_customer(db, phone=from_number)
                upsert_vehicle_from_vin(db, customer, vin_info)

            msg = [
                "VIN decoded:",
                f"VIN: {vin_info.get('vin')}",
                f"Year: {vin_info.get('year')}",
                f"Make: {vin_info.get('make')}",
                f"Model: {vin_info.get('model')}",
            ]
            reply.message("\n".join(msg))
            return Response(content=str(reply), media_type="application/xml")

        # E) Booking request (simplified format):
        # "Book Full Name, email@example.com, Nov 28 9am"
        if lower_body.startswith("book"):
            # Remove the word "book" and split the rest by comma
            remainder = body[4:].strip()
            parts = [p.strip() for p in remainder.split(",") if p.strip()]
            if len(parts) < 3:
                reply.message(
                    "To book, please reply like this:\n"
                    "Book Full Name, email@example.com, Nov 28 9am"
                )
                return Response(content=str(reply), media_type="application/xml")

            full_name = parts[0]
            email = parts[1]
            dt_str = ",".join(parts[2:])  # in case date contains comma

            try:
                appt_dt = parse_appointment_datetime(dt_str)
            except Exception:
                reply.message(
                    "I couldn't read that date/time. Please try something like:\n"
                    "Book John Smith, john@email.com, Nov 28 9am"
                )
                return Response(content=str(reply), media_type="application/xml")

            now_utc = datetime.utcnow()
            if appt_dt <= now_utc + timedelta(minutes=MIN_LEAD_MINUTES):
                with db_session() as db:
                    suggestions = find_next_available_slots(
                        db, shop, now_utc + timedelta(minutes=MIN_LEAD_MINUTES)
                    )
                if suggestions:
                    lines = [
                        "That time is too soon or already past.",
                        "",
                        "Here are some next available times:",
                    ]
                    for s in suggestions:
                        lines.append(f"- {format_slot(s)}")
                    lines.append("")
                    lines.append(
                        "To book one of these, reply like:\n"
                        "Book Full Name, email@example.com, Wed Nov 27 3pm"
                    )
                    reply.message("\n".join(lines))
                else:
                    reply.message(
                        "That time is too soon or in the past. Please pick a future time during open hours."
                    )
                return Response(content=str(reply), media_type="application/xml")

            # Check business hours and availability
            with db_session() as db:
                hours = get_shop_hours_for_day(shop, appt_dt)
                if not hours:
                    # Closed that day
                    suggestions = find_next_available_slots(db, shop, appt_dt)
                    lines = [
                        "The shop is closed at that date/time.",
                    ]
                    if suggestions:
                        lines.append("")
                        lines.append("Here are some next available times:")
                        for s in suggestions:
                            lines.append(f"- {format_slot(s)}")
                        lines.append("")
                        lines.append(
                            "To book one of these, reply like:\n"
                            "Book Full Name, email@example.com, Thu Nov 28 9am"
                        )
                    else:
                        lines.append(
                            "Please choose another day during our open hours (Mon–Sat)."
                        )
                    reply.message("\n".join(lines))
                    return Response(content=str(reply), media_type="application/xml")

                if not is_within_shop_hours(shop, appt_dt):
                    suggestions = find_next_available_slots(db, shop, appt_dt)
                    lines = [
                        "That time is outside shop hours.",
                    ]
                    if suggestions:
                        lines.append("")
                        lines.append("Here are some times within open hours:")
                        for s in suggestions:
                            lines.append(f"- {format_slot(s)}")
                        lines.append("")
                        lines.append(
                            "To book one of these, reply like:\n"
                            "Book Full Name, email@example.com, Fri Nov 29 10am"
                        )
                    else:
                        lines.append(
                            "Please select a time within our open hours (Mon–Sat)."
                        )
                    reply.message("\n".join(lines))
                    return Response(content=str(reply), media_type="application/xml")

                if not is_slot_available(db, shop, appt_dt):
                    suggestions = find_next_available_slots(
                        db, shop, appt_dt + timedelta(minutes=1)
                    )
                    lines = [
                        "That time is already booked.",
                    ]
                    if suggestions:
                        lines.append("")
                        lines.append("Here are some other available times:")
                        for s in suggestions:
                            lines.append(f"- {format_slot(s)}")
                        lines.append("")
                        lines.append(
                            "To book one of these, reply like:\n"
                            "Book Full Name, email@example.com, Sat Nov 30 11am"
                        )
                    else:
                        lines.append(
                            "Please choose another time during our open hours (Mon–Sat)."
                        )
                    reply.message("\n".join(lines))
                    return Response(content=str(reply), media_type="application/xml")

                # Slot is valid and free -> save + calendar
                customer = get_or_create_customer(
                    db, phone=from_number, full_name=full_name, email=email
                )

                success, msg, event_id = create_calendar_event_for_booking(
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

            if success:
                reply.message(
                    "Your repair appointment has been booked.\n\n"
                    f"{msg}\n\n"
                    f"Time: {format_slot(appt_dt)}"
                )
            else:
                reply.message(
                    "I saved your booking details, but couldn't add it to the calendar automatically:\n"
                    f"{msg}\n\nThe shop will follow up to confirm your time.\n\n"
                    f"Requested time: {format_slot(appt_dt)}"
                )

            SESSIONS[key] = session
            return Response(content=str(reply), media_type="application/xml")

        # F) Default instructions
        instructions = (
            f"Hi from {shop.name}!\n\n"
            "To get an AI-powered damage estimate:\n"
            "1) Send 1–3 clear photos of the damaged area.\n"
            "2) I’ll send a quick AI Pre-Scan.\n"
            "3) Reply 1 if it looks right, or 2 if it’s off.\n"
            "4) Then I'll send your full Ontario 2025 cost estimate.\n\n"
            "Optional:\n"
            "- Text your 17-character VIN to decode your vehicle details.\n"
            "- After your estimate, book a repair by replying:\n"
            "  Book Full Name, email@example.com, Nov 28 9am"
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
# Admin: list shops + example webhook URLs
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

# ============================================================
# Health check
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI damage estimator with Level-C pricing, DB, fusion, VIN & booking is running",
    }
