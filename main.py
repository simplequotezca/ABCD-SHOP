from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

import datetime
import os
import json
import httpx
from typing import Dict, Optional, List
import re
import uuid
import base64

from twilio.rest import Client

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================================================
# ENV + DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. On Railway, attach Postgres then copy the "
        "full connection URL into a DATABASE_URL variable."
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

app = FastAPI()

# ============================================================
# TWILIO CLIENT (for downloading media)
# ============================================================

twilio_client: Optional[Client] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def download_twilio_image(media_url: str) -> bytes:
    """
    Download an MMS image from Twilio using the official client so
    authentication is handled correctly.
    """
    if not twilio_client:
        raise RuntimeError("Twilio client not configured")

    resp = twilio_client.request("GET", media_url)
    if resp.status_code != 200:
        raise Exception(
            f"Error downloading media: {resp.status_code} {resp.text}"
        )

    return resp.content


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    """
    Convert raw image bytes to a base64 data URL that OpenAI can consume.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
    # ============================================================
# SUPER PROMPT v2.0 + VALIDATION LAYER
# ============================================================

ALLOWED_PANELS = {
    "front bumper", "rear bumper",
    "hood", "trunk",
    "left fender", "right fender",
    "left front door", "right front door",
    "left rear door", "right rear door",
    "left quarter panel", "right quarter panel",
    "rocker panel", "roof",
    "rim/wheel", "tire",
    "headlight", "tail light",
}

ALLOWED_DAMAGE_TYPES = {
    "scratch", "paint scuff", "paint chip",
    "dent", "crease", "crack",
    "curb rash", "gouge", "bent rim",
    "bumper deformation", "misalignment",
}

ALLOWED_SEVERITY = {"Minor", "Moderate", "Severe", "unknown"}
ALLOWED_TIERS = {"Basic Cosmetic", "Standard Repair", "Premium / Extensive", "unknown"}


def compute_pricing_tier(min_cost: float, max_cost: float) -> str:
    avg = (min_cost + max_cost) / 2.0
    if max_cost <= 0:
        return "unknown"
    if avg <= 600:
        return "Basic Cosmetic"
    if avg <= 2500:
        return "Standard Repair"
    return "Premium / Extensive"


def validate_and_normalize_estimate(raw: dict) -> dict:
    """
    Normalizes and protects against AI hallucinations or malformed JSON.
    """
    result: dict = {}

    # Severity
    severity = str(raw.get("severity", "unknown"))
    if severity not in ALLOWED_SEVERITY:
        severity = "unknown"
    result["severity"] = severity

    # Panels
    panels = raw.get("panels") or []
    if not isinstance(panels, list):
        panels = []
    clean_panels = [
        p for p in panels if isinstance(p, str) and p in ALLOWED_PANELS
    ]
    result["panels"] = clean_panels

    # Damage types
    damage_types = raw.get("damage_types") or []
    if not isinstance(damage_types, list):
        damage_types = []
    clean_damage = [
        d for d in damage_types
        if isinstance(d, str) and d in ALLOWED_DAMAGE_TYPES
    ]
    result["damage_types"] = clean_damage

    # Costs
    try:
        min_cost = float(raw.get("estimated_cost_min", 0) or 0)
    except Exception:
        min_cost = 0.0
    try:
        max_cost = float(raw.get("estimated_cost_max", 0) or 0)
    except Exception:
        max_cost = 0.0

    if max_cost < min_cost:
        min_cost, max_cost = max_cost, min_cost

    # Sanity caps
    if max_cost - min_cost > 20000:
        mid = (min_cost + max_cost) / 2.0
        min_cost, max_cost = mid - 5000, mid + 5000

    min_cost = max(0.0, round(min_cost))
    max_cost = max(min_cost, round(max_cost))

    # If unknown severity, zero-out costs
    if severity == "unknown":
        min_cost, max_cost = 0.0, 0.0

    result["estimated_cost_min"] = min_cost
    result["estimated_cost_max"] = max_cost

    # Confidence
    try:
        confidence = float(raw.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    result["confidence"] = max(0, min(1, confidence))

    # Pricing tier
    tier = raw.get("pricing_tier", "unknown")
    if tier not in ALLOWED_TIERS:
        tier = compute_pricing_tier(min_cost, max_cost)
    result["pricing_tier"] = tier

    return result


# ============================================================
# SUPER-PROMPT v2.0
# ============================================================
system_prompt = """
You are an advanced automotive damage estimator AI used by collision and wheel repair shops in Ontario, Canada in the year 2025.

You analyze up to 5 photos of vehicle damage and return STRICT JSON ONLY. Never output text outside JSON.

=====================
RULES:
=====================
- DO NOT guess parts that are not visible.
- DO NOT guess vehicle make/model/year.
- DO NOT hallucinate panels.
- IF image unclear: severity="unknown", costs=0, tier="unknown", confidence<=0.2
- Use only allowed panel names and damage types.
- Never invent new fields.
- Multi-image: consider all angles, choose the worst visible damage.

=====================
ONTARIO 2025 REPAIR COSTS:
=====================
Cosmetic scratches: 150–450  
Minor dents: 200–600  
Rim curb rash: 120–250  
Rim refinishing: 200–350  
Moderate dents paint damage: 400–900  
Bumper scuff+deformation: 600–1200  
Rim bend repair: 220–380  
Cracked rim: 350–600+  
Bumper replacement: 750–1800  

=====================
SEVERITY:
=====================
Minor = cosmetic  
Moderate = structural-looking but drivable  
Severe = major deformation, cracks, broken components  

=====================
PRICING TIER:
=====================
<=600 → Basic Cosmetic  
601–2500 → Standard Repair  
>2500 → Premium / Extensive  

=====================
FINAL JSON KEYS (MANDATORY):
=====================
{
  "severity": "...",
  "panels": [...],
  "damage_types": [...],
  "estimated_cost_min": number,
  "estimated_cost_max": number,
  "confidence": number,
  "pricing_tier": "..."
}
"""


# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """Parse SHOPS_JSON into token->ShopConfig."""
    if not SHOPS_JSON:
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        return {default.webhook_token: default}

    try:
        data = json.loads(SHOPS_JSON)
        shops: Dict[str, ShopConfig] = {}
        for s in data:
            shop = ShopConfig(**s)
            shops[shop.webhook_token] = shop
        return shops
    except Exception:
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        return {default.webhook_token: default}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}
# ============================================================
# DATABASE MODELS
# ============================================================

class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    shop_id = Column(String, index=True)
    customer_phone = Column(String, index=True)
    severity = Column(String)
    damage_areas = Column(Text)          # comma-separated panels
    damage_types = Column(Text)          # comma-separated damage types
    recommended_repairs = Column(Text)   # kept for compatibility
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String, nullable=True)
    customer_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS: IMAGES + VIN
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    urls = []
    index = 0
    while True:
        key = f"MediaUrl{index}"
        url = form.get(key)
        if not url:
            break
        urls.append(url)
        index += 1
    return urls


def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    match = VIN_PATTERN.search(text.upper())
    return match.group(1) if match else None


# ============================================================
# AI DAMAGE ESTIMATION (SUPER ESTIMATOR v2.0)
# ============================================================

async def estimate_damage_from_images(
    image_urls: List[str], 
    vin: Optional[str], 
    shop: ShopConfig,
) -> dict:
    """
    Full AI estimator with:
    - multi-image support (up to 5)
    - strict JSON schema
    - fraud/unclear detection
    - pricing tiers
    - normalization layer
    """
    if not OPENAI_API_KEY:
        return {
            "severity": "unknown",
            "panels": [],
            "damage_types": [],
            "estimated_cost_min": 0,
            "estimated_cost_max": 0,
            "confidence": 0.1,
            "pricing_tier": "unknown",
            "customer_summary": (
                "Our AI estimator is temporarily unavailable. "
                "A technician will provide a manual review."
            )
        }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build content payload with up to 5 images
    content = []
    intro = (
        f"Estimate damage for {shop.name} in Ontario. "
        "Follow system rules strictly."
    )
    if vin:
        intro += f" VIN: {vin}"
    content.append({"type": "text", "text": intro})

    usable_images = image_urls[:5]

    for url in usable_images:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img_bytes = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img_bytes)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except Exception as e:
                print("Twilio media error:", e)
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()

        raw_str = resp.json()["choices"][0]["message"]["content"]
        raw = json.loads(raw_str)

        normalized = validate_and_normalize_estimate(raw)

        # Build customer summary
        sev = normalized["severity"]
        min_c = normalized["estimated_cost_min"]
        max_c = normalized["estimated_cost_max"]
        tier = normalized["pricing_tier"]
        conf = normalized["confidence"]
        panels = ", ".join(normalized["panels"]) or "no visible panels"
        damages = ", ".join(normalized["damage_types"]) or "no clear damage type"

        if sev == "unknown":
            summary = (
                "The photos are unclear or do not show identifiable damage. "
                "An in-person inspection is recommended to provide an accurate estimate."
            )
        else:
            summary = (
                f"The AI detected {sev.lower()} damage affecting: {panels}. "
                f"Visible issues: {damages}. "
                f"Estimated Ontario repair cost: ${min_c:,.0f}–${max_c:,.0f} ({tier}). "
                f"Confidence level: {conf:.2f}. "
                f"An in-person inspection may adjust the final cost."
            )

        normalized["customer_summary"] = summary
        return normalized

    except Exception as e:
        print("AI estimator error:", e)
        return {
            "severity": "unknown",
            "panels": [],
            "damage_types": [],
            "estimated_cost_min": 0,
            "estimated_cost_max": 0,
            "confidence": 0.1,
            "pricing_tier": "unknown",
            "customer_summary": (
                "We had trouble analyzing the images. "
                "A technician will review them manually."
            )
    }
# ============================================================
# SAVE ESTIMATE TO DATABASE
# ============================================================

def save_estimate_to_db(shop: ShopConfig, phone: str, vin: Optional[str], result: dict) -> str:
    """
    Saves the upgraded estimator output into DB using existing columns.
    - recommended_repairs is preserved but left blank (Option C).
    """
    db = SessionLocal()
    try:
        est = Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=result.get("severity"),
            damage_areas=", ".join(result.get("panels", [])),
            damage_types=", ".join(result.get("damage_types", [])),
            recommended_repairs="",  # kept for compatibility, not used anymore
            min_cost=result.get("estimated_cost_min"),
            max_cost=result.get("estimated_cost_max"),
            confidence=result.get("confidence"),
            vin=vin,
            customer_summary=result.get("customer_summary"),
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


# ============================================================
# ADMIN AUTH
# ============================================================

def require_admin(request: Request):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ============================================================
# APPOINTMENT SLOT GENERATOR
# ============================================================

def get_appointment_slots(n: int = 3) -> List[datetime.datetime]:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)

    hours = [9, 11, 14, 16]  # 9am, 11am, 2pm, 4pm
    slots = []

    for hour in hours:
        t = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
        if t > now:
            slots.append(t)

    return slots[:n]


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Auto-shop backend running"}


# ============================================================
# TWILIO SMS WEBHOOK — MAIN ENTRY
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    """
    Twilio SMS webhook.
    1. If customer replies 1/2/3 → book appointment.
    2. If customer sends images → run AI estimator.
    3. If no images → send instructions.
    """
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    # Track sessions
    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # ========================================================
    # STEP 1: Booking reply (user sends 1, 2, or 3)
    # ========================================================
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]
            msg = [
                f"You're booked at {shop.name}.",
                "",
                "Appointment time:",
                chosen.strftime("%a %b %d at %I:%M %p"),
                "",
                "If you need to change this time, reply 'Change'."
            ]
            reply.message("\n".join(msg))

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # ========================================================
    # STEP 2: Photo-based AI estimate
    # ========================================================
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        min_cost = result["estimated_cost_min"]
        max_cost = result["estimated_cost_max"]
        tier = result["pricing_tier"]
        panels = ", ".join(result["panels"]) or "No clear panels visible"
        types = ", ".join(result["damage_types"]) or "Not clearly identifiable"
        summary = result["customer_summary"]

        if max_cost > 0:
            cost_range = f"${min_cost:,.0f} – ${max_cost:,.0f}"
        else:
            cost_range = "N/A (photos unclear)"

        # Appointment slots
        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        msg_lines = [
            f"AI Damage Estimate for {shop.name}",
            "",
            f"Severity: {severity}",
            f"Estimated Cost (Ontario 2025): {cost_range}",
            f"Damage Areas: {panels}",
            f"Damage Types: {types}",
            f"Tier: {tier}",
            "",
            summary,
            "",
            f"Estimate ID (internal): {estimate_id}",
            "",
            "Reply with a number to book an in-person estimate:",
        ]

        for i, s in enumerate(slots, 1):
            msg_lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(msg_lines))
        return Response(content=str(reply), media_type="application/xml")

    # ========================================================
    # STEP 3: No images → send instructions
    # ========================================================
    intro_msg = [
        f"Thanks for messaging {shop.name}.",
        "",
        "To get an AI-powered pre-estimate:",
        "- Send 1–5 clear photos of the damage",
        "- Optional: include your 17-digit VIN",
    ]
    reply.message("\n".join(intro_msg))
    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# ADMIN API (READ-ONLY)
# ============================================================

@app.get("/admin/estimates")
def list_estimates(
    request: Request,
    shop_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
):
    require_admin(request)
    db = SessionLocal()
    try:
        q = db.query(Estimate)
        if shop_id:
            q = q.filter(Estimate.shop_id == shop_id)
        q = q.order_by(Estimate.created_at.desc()).offset(skip).limit(limit)

        return [
            {
                "id": e.id,
                "shop_id": e.shop_id,
                "customer_phone": e.customer_phone,
                "severity": e.severity,
                "min_cost": e.min_cost,
                "max_cost": e.max_cost,
                "created_at": e.created_at.isoformat(),
            }
            for e in q.all()
        ]
    finally:
        db.close()


@app.get("/admin/estimates/{estimate_id}")
def get_estimate(estimate_id: str, request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        e = db.query(Estimate).filter(Estimate.id == estimate_id).first()

        if not e:
            raise HTTPException(status_code=404, detail="Estimate not found")

        return {
            "id": e.id,
            "shop_id": e.shop_id,
            "customer_phone": e.customer_phone,
            "severity": e.severity,
            "damage_areas": e.damage_areas,
            "damage_types": e.damage_types,
            "recommended_repairs": e.recommended_repairs,
            "min_cost": e.min_cost,
            "max_cost": e.max_cost,
            "confidence": e.confidence,
            "vin": e.vin,
            "customer_summary": e.customer_summary,
            "created_at": e.created_at.isoformat(),
        }

    finally:
        db.close()
