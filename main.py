from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import datetime
import os
import json
import httpx
import asyncio
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
    raise RuntimeError("DATABASE_URL is not set. Attach Postgres in Railway first.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

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
        shops = {}
        for s in data:
            shop = ShopConfig(**s)
            shops[shop.webhook_token] = shop
        return shops
    except Exception:
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
# HELPERS — VIN + IMAGES
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    urls = []
    i = 0
    while True:
        key = f"MediaUrl{i}"
        url = form.get(key)
        if not url:
            break
        urls.append(url)
        i += 1
    return urls[:2]  # LIMIT TO TWO IMAGES


def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    match = VIN_PATTERN.search(text.upper())
    return match.group(1) if match else None


# ============================================================
# GPT-4o-mini DAMAGE ESTIMATION + RETRIES
# ============================================================

async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig) -> dict:

    if not OPENAI_API_KEY:
        return fallback_result(vin_used=False)

    system_prompt = """
You are an Ontario (Canada, 2025) certified auto-body estimator.
Output ONLY JSON.
""".strip()

    content = [
        {"type": "text", "text": "Analyze the vehicle damage images."}
    ]

    if vin:
        content[0]["text"] += f" VIN: {vin}"

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }

    async def call_openai():
        async with httpx.AsyncClient(timeout=45) as client:
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            res.raise_for_status()
            return res.json()

    # Retry up to 3 times
    for attempt in range(3):
        try:
            data = await call_openai()
            raw = data["choices"][0]["message"]["content"]
            result = json.loads(raw)

            # sanity defaults
            result.setdefault("severity", "Moderate")
            result.setdefault("damage_areas", [])
            result.setdefault("damage_types", [])
            result.setdefault("recommended_repairs", [])
            result.setdefault("min_cost", 600)
            result.setdefault("max_cost", 1500)
            result.setdefault("confidence", 0.75)
            result.setdefault("vin_used", bool(vin))

            return result

        except Exception as e:
            print(f"[AI ERROR] Attempt {attempt+1}: {e}")
            if attempt < 2:
                await asyncio.sleep(1.5)
            else:
                return fallback_result(vin_used=bool(vin))


def fallback_result(vin_used=False):
    return {
        "severity": "Moderate",
        "damage_areas": [],
        "damage_types": [],
        "recommended_repairs": [],
        "min_cost": 600,
        "max_cost": 1500,
        "confidence": 0.5,
        "vin_used": vin_used,
    }


# ============================================================
# APPOINTMENT SLOTS
# ============================================================

def get_appointment_slots(n=3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hrs = [9, 11, 14]

    slots = []
    for h in hrs:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ============================================================
# SAVE ESTIMATE
# ============================================================

def save_estimate_to_db(shop: ShopConfig, phone: str, vin: Optional[str], r: dict) -> str:
    db = SessionLocal()
    try:
        est = Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=r["severity"],
            damage_areas=", ".join(r["damage_areas"]),
            damage_types=", ".join(r["damage_types"]),
            recommended_repairs=", ".join(r["recommended_repairs"]),
            min_cost=r["min_cost"],
            max_cost=r["max_cost"],
            confidence=r["confidence"],
            vin=vin,
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):

    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From")

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    # SESSION KEY
    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # Booking confirmation
    if session and session.get("awaiting") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]
            reply.message(
                f"You're booked at {shop.name} on "
                f"{chosen.strftime('%a %b %d at %I:%M %p')}."
            )
            session["awaiting"] = False
            return Response(str(reply), media_type="application/xml")

    # Images detected → run AI
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        est_id = save_estimate_to_db(shop, from_number, vin, result)

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting": True, "slots": slots}

        lines = [
            f"AI Damage Estimate for {shop.name}",
            "",
            f"Severity: {result['severity']}",
            f"Estimated Cost (Ontario 2025): ${result['min_cost']:,} – ${result['max_cost']:,}",
            "",
            "Detected Panels:",
            ", ".join(result["damage_areas"]) or "-",
            "",
            "Damage Types:",
            ", ".join(result["damage_types"]) or "-",
            "",
            f"Estimate ID:\n{est_id}",
            "",
            "Reply with a number to book an in-person estimate:",
        ]

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(str(reply), media_type="application/xml")

    # No images → intro
    intro = [
        f"Thanks for messaging {shop.name}.",
        "",
        "To get an AI-powered estimate:",
        "- Send 1–2 clear photos of the damage",
        "- Optional: include your VIN",
    ]
    reply.message("\n".join(intro))
    return Response(str(reply), media_type="application/xml")
