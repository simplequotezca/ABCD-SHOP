from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

import datetime
import os
import json
import httpx
import uuid
import re

from typing import Dict, Optional, List

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set — attach Railway Postgres and set DATABASE_URL.")


# ============================================================
# DATABASE SETUP
# ============================================================

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ============================================================
# FASTAPI INSTANCE
# ============================================================

app = FastAPI()


# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """Parse SHOPS_JSON → dict[token] = ShopConfig"""
    try:
        data = json.loads(SHOPS_JSON)
        shops = {s["webhook_token"]: ShopConfig(**s) for s in data}
        return shops
    except Exception:
        # fallback default shop
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo_token")
        return {default.webhook_token: default}


SHOPS_BY_TOKEN = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing shop token")
    return SHOPS_BY_TOKEN[token]


# ============================================================
# DATABASE MODELS
# ============================================================

class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    shop_id = Column(String)
    customer_phone = Column(String)
    severity = Column(String)
    damage_areas = Column(Text)
    damage_types = Column(Text)
    recommended_repairs = Column(Text)
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS: EXTRACT IMAGES + VIN
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")

def extract_image_urls(form) -> List[str]:
    urls = []
    i = 0
    while True:
        url = form.get(f"MediaUrl{i}")
        if not url:
            break
        urls.append(url)
        i += 1
    return urls

def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    match = VIN_PATTERN.search(text.upper())
    if match:
        return match.group(1)
    return None


# ============================================================
# AI DAMAGE ESTIMATOR (gpt-4o-mini vision)
# ============================================================

async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig):
    """High-quality estimate using gpt-4o-mini."""
    if not OPENAI_API_KEY:
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.5,
            "vin_used": False,
        }

    model = "gpt-4o-mini"

    system_prompt = """
You are a certified Ontario auto-body estimator (2025).
Analyze damage, extract specific panels, damage types, and Ontario-calibrated CAD pricing.
Return ONLY valid JSON.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build user content for vision input
    content = [{"type": "text", "text": "Analyze these vehicle damage images."}]
    if vin:
        content.append({"type": "text", "text": f"VIN: {vin}"})

    for url in image_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()

        result = json.loads(resp.json()["choices"][0]["message"]["content"])

        # Sanity defaults
        result.setdefault("severity", "Moderate")
        result.setdefault("damage_areas", [])
        result.setdefault("damage_types", [])
        result.setdefault("recommended_repairs", [])
        result.setdefault("min_cost", 600)
        result.setdefault("max_cost", 1500)
        result.setdefault("confidence", 0.7)
        result.setdefault("vin_used", bool(vin))

        return result

    except Exception as e:
        print("AI error:", e)
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.5,
            "vin_used": bool(vin),
        }


# ============================================================
# SAVE ESTIMATE
# ============================================================

def save_estimate_to_db(shop: ShopConfig, phone: str, vin: Optional[str], result: dict) -> str:
    db = SessionLocal()
    try:
        est = Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=result["severity"],
            damage_areas=", ".join(result["damage_areas"]),
            damage_types=", ".join(result["damage_types"]),
            recommended_repairs=", ".join(result["recommended_repairs"]),
            min_cost=result["min_cost"],
            max_cost=result["max_cost"],
            confidence=result["confidence"],
            vin=vin
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


# ============================================================
# APPOINTMENT SLOTS
# ============================================================

def get_appointment_slots(n: int = 3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    slots = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Auto-shop backend running"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):

    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # Booking selection
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slot = session["slots"][idx]
        reply.message(
            f"Your appointment at {shop.name} is booked for "
            f"{slot.strftime('%a %b %d at %I:%M %p')}."
        )
        session["awaiting_time"] = False
        SESSIONS[session_key] = session
        return Response(content=str(reply), media_type="application/xml")

    # AI Estimate
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        min_cost = result["min_cost"]
        max_cost = result["max_cost"]

        areas = ", ".join(result["damage_areas"]) or "Detected panels"
        types = ", ".join(result["damage_types"]) or "Detected damage types"

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"AI Damage Estimate for {shop.name}",
            f"Severity: {severity}",
            f"Estimated Cost: ${min_cost:,.0f} - ${max_cost:,.0f}",
            f"Damage Areas: {areas}",
            f"Damage Types: {types}",
            f"Estimate ID: {estimate_id}",
            "",
            "Reply with a number to book an appointment:",
        ]

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # Default message
    reply.message(
        f"Welcome to {shop.name}!\n\n"
        "Send 1–5 photos of the vehicle damage.\n"
        "Optional: Include your 17-character VIN."
    )
    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/admin/estimates")
def list_estimates(request: Request):
    if request.headers.get("x-api-key") != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    db = SessionLocal()
    rows = db.query(Estimate).order_by(Estimate.created_at.desc()).limit(50)
    out = [
        {
            "id": e.id,
            "shop_id": e.shop_id,
            "phone": e.customer_phone,
            "severity": e.severity,
            "min_cost": e.min_cost,
            "max_cost": e.max_cost,
            "created_at": e.created_at.isoformat(),
        }
        for e in rows
    ]
    db.close()
    return out
