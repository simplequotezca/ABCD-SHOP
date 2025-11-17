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
# ENVIRONMENT
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing in Railway!")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

app = FastAPI()

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")

# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops():
    if not SHOPS_JSON:
        return {"shop_sj_84k2p1": ShopConfig(id="sj_auto_body", name="SJ Auto Body", webhook_token="shop_sj_84k2p1")}

    data = json.loads(SHOPS_JSON)
    shops = {}
    for s in data:
        shop = ShopConfig(**s)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid shop token")
    return SHOPS_BY_TOKEN[token]


# ============================================================
# DATABASE MODEL
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
    vin = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS
# ============================================================

def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    m = VIN_PATTERN.search(text.upper())
    return m.group(1) if m else None


def extract_image_urls(form) -> List[str]:
    urls = []
    for i in range(5):
        u = form.get(f"MediaUrl{i}")
        if u:
            urls.append(u)
    return urls[:2]     # limit to 2 images for 4o-mini stability


# ============================================================
# AI DAMAGE ESTIMATOR
# ============================================================

async def analyze_damage(image_urls, vin):
    """
    Calls GPT-4o-mini vision with stable retries.
    """

    system_prompt = """
You are a certified Ontario auto body damage estimator.
Provide precise and specific analysis. Output ONLY JSON:

{
 "severity": "",
 "damage_areas": [],
 "damage_types": [],
 "recommended_repairs": [],
 "min_cost": number,
 "max_cost": number,
 "confidence": number
}
"""

    user_content = [{"type": "text", "text": "Analyze the vehicle damage."}]
    if vin:
        user_content.append({"type": "text", "text": f"VIN: {vin}"})

    for url in image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=40) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
            r.raise_for_status()

            raw = r.json()["choices"][0]["message"]["content"]
            data = json.loads(raw)

            # sanitize
            data.setdefault("severity", "Moderate")
            data.setdefault("damage_areas", [])
            data.setdefault("damage_types", [])
            data.setdefault("recommended_repairs", [])
            data.setdefault("min_cost", 600)
            data.setdefault("max_cost", 1500)
            data.setdefault("confidence", 0.7)

            return data

        except Exception as e:
            print("AI error attempt", attempt + 1, e)

    # If all fails → fallback
    return {
        "severity": "Moderate",
        "damage_areas": [],
        "damage_types": [],
        "recommended_repairs": [],
        "min_cost": 600,
        "max_cost": 1500,
        "confidence": 0.5
    }


# ============================================================
# SAVE ESTIMATE
# ============================================================

def save_estimate(shop, phone, vin, result):
    db = SessionLocal()
    e = Estimate(
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
    db.add(e)
    db.commit()
    db.refresh(e)
    db.close()
    return e.id


# ============================================================
# APPOINTMENT SLOTS
# ============================================================

def get_slots():
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    times = [9, 11, 14]
    slots = []
    for t in times:
        dt = tomorrow.replace(hour=t, minute=0, second=0)
        slots.append(dt)
    return slots


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {"message": "Backend running"}


@app.post("/sms-webhook")
async def webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    form = await request.form()
    msg = (form.get("Body") or "").strip()
    phone = form.get("From")
    images = extract_image_urls(form)
    vin = extract_vin(msg)

    reply = MessagingResponse()

    # If user is choosing a time
    session_key = f"{shop.id}:{phone}"
    session = SESSIONS.get(session_key)

    if session and session.get("waiting_for_time"):
        if msg in {"1", "2", "3"}:
            chosen = session["slots"][int(msg)-1]
            reply.message(
                f"You're booked at {shop.name} on {chosen.strftime('%a %b %d at %I:%M %p')}."
            )
            session["waiting_for_time"] = False
            return Response(content=str(reply), media_type="application/xml")

    # If user sent images → run AI
    if images:
        result = await analyze_damage(images, vin)
        estimate_id = save_estimate(shop, phone, vin, result)

        slots = get_slots()
        SESSIONS[session_key] = {"waiting_for_time": True, "slots": slots}

        text = f"""AI Damage Estimate for {shop.name}

Severity: {result['severity']}
Estimated Cost (Ontario 2025): ${result['min_cost']:,} – ${result['max_cost']:,}

Detected Panels:
- {", ".join(result['damage_areas']) if result['damage_areas'] else "-"}

Damage Types:
- {", ".join(result['damage_types']) if result['damage_types'] else "-"}

Estimate ID:
{estimate_id}

Reply with a number to book an in-person estimate:
1) {slots[0].strftime('%a %b %d at %I:%M %p')}
2) {slots[1].strftime('%a %b %d at %I:%M %p')}
3) {slots[2].strftime('%a %b %d at %I:%M %p')}
"""

        reply.message(text)
        return Response(content=str(reply), media_type="application/xml")

    # Otherwise → onboarding
    reply.message(
        f"Thanks for messaging {shop.name}!\n\n"
        "To get an AI-powered repair estimate:\n"
        "• Send 1–2 photos of the damage\n"
        "• Optional: include your 17-digit VIN"
    )
    return Response(content=str(reply), media_type="application/xml")
