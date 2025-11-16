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

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base


# ============================================================
# ENV + DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

app = FastAPI()


# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    if not SHOPS_JSON:
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo_token")
        return {default.webhook_token: default}

    try:
        data = json.loads(SHOPS_JSON)
        shops = {s["webhook_token"]: ShopConfig(**s) for s in data}
        return shops
    except Exception as e:
        print("Failed to parse SHOPS_JSON:", e)
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo_token")
        return {default.webhook_token: default}


SHOPS_BY_TOKEN = load_shops()
SESSIONS = {}


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
    shop_id = Column(String, index=True)
    customer_phone = Column(String, index=True)
    severity = Column(String)
    damage_areas = Column(Text)
    damage_types = Column(Text)
    recommended_repairs = Column(Text)
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    urls = []
    i = 0
    while form.get(f"MediaUrl{i}"):
        urls.append(form.get(f"MediaUrl{i}"))
        i += 1
    return urls


def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    m = VIN_PATTERN.search(text.upper())
    return m.group(1) if m else None


# ============================================================
# AI DAMAGE ESTIMATION
# ============================================================

async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig) -> dict:
    if not OPENAI_API_KEY:
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600, "max_cost": 1500,
            "confidence": 0.5,
            "vin_used": False
        }

    system_prompt = """
You are an Ontario 2025 certified auto damage estimator...
(omitted here for brevity — keep your original long prompt)
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    content = [{"type": "text", "text": "Analyze these photos."}]
    if vin:
        content[0]["text"] += f" VIN: {vin}"

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers
            )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(raw)
        data.setdefault("vin_used", bool(vin))

        return data

    except Exception as e:
        print("AI error:", e)
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600, "max_cost": 1500,
            "confidence": 0.5,
            "vin_used": bool(vin)
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
            vin=vin,
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


def get_appointment_slots(n=3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    return [
        tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in [9, 11, 14, 16]
    ][:n]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()
    session_key = f"{shop.id}:{from_number}"

    # Booking response
    session = SESSIONS.get(session_key)
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slot = session["slots"][idx]

        reply.message(
            f"You're booked at {shop.name} for {slot.strftime('%a %b %d at %I:%M %p')}."
        )
        session["awaiting_time"] = False
        return Response(content=str(reply), media_type="application/xml")

    # AI estimate
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        cost_range = f"${result['min_cost']:,.0f} – ${result['max_cost']:,.0f}"

        areas = result["damage_areas"]
        types = result["damage_types"]

        # Clean SMS formatting
        sms = f"""
AI Damage Estimate for {shop.name}

Severity: {severity}
Estimated Cost (Ontario 2025): {cost_range}

Detected Panels:
- {chr(10).join(areas)}

Damage Types:
- {chr(10).join(types)}

Estimate ID:
{estimate_id}
""".strip()

        if vin and result.get("vin_used"):
            sms += f"\n\nVIN used for calibration: {vin}"

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        sms += "\n\nReply with a number to book an in-person estimate:\n"
        for i, s in enumerate(slots, 1):
            sms += f"{i}) {s.strftime('%a %b %d at %I:%M %p')}\n"

        reply.message(sms)
        return Response(content=str(reply), media_type="application/xml")

    # Default message
    reply.message(
        f"Thanks for messaging {shop.name}.\n\n"
        "To get an AI-powered pre-estimate:\n"
        "- Send 1–5 clear photos of the damage\n"
        "- Optional: include your 17-character VIN"
    )
    return Response(content=str(reply), media_type="application/xml")
