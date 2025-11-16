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

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================================================
# ENV + DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

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
# SHOP CONFIG
# ============================================================


class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """
    Parse SHOPS_JSON into a token -> ShopConfig map.

    Example SHOPS_JSON:

    [
      {
        "id": "sj_auto_body",
        "name": "SJ Auto Body",
        "webhook_token": "shop_sj_84k2p1"
      }
    ]
    """
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        default = ShopConfig(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )
        return {default.webhook_token: default}

    try:
        data = json.loads(raw)
        shops: Dict[str, ShopConfig] = {}
        for item in data:
            shop = ShopConfig(**item)
            shops[shop.webhook_token] = shop
        return shops
    except Exception as e:
        print("Failed to parse SHOPS_JSON:", e)
        default = ShopConfig(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )
        return {default.webhook_token: default}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    """
    Pick shop based on ?token= in the Twilio webhook URL.

    Example Twilio URL:
    https://your-service.up.railway.app/sms-webhook?token=shop_sj_84k2p1
    """
    if not SHOPS_BY_TOKEN:
        return ShopConfig(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )

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
    damage_areas = Column(Text)          # comma-separated
    damage_types = Column(Text)          # comma-separated
    recommended_repairs = Column(Text)   # comma-separated
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS: IMAGES + VIN
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    """Collect all MediaUrlN fields from the Twilio form."""
    urls: List[str] = []
    i = 0
    while True:
        key = f"MediaUrl{i}"
        url = form.get(key)
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
# AI DAMAGE ESTIMATION (ONTARIO 2025, MULTI-IMAGE)
# ============================================================


async def estimate_damage_from_images(
    image_urls: List[str],
    vin: Optional[str],
    shop: ShopConfig,
) -> dict:
    """
    Call OpenAI vision model to estimate damage and Ontario 2025 cost range.
    Returns a dict with keys:
    severity, damage_areas, damage_types, recommended_repairs,
    min_cost, max_cost, confidence, vin_used
    """
    if not OPENAI_API_KEY:
        # Fallback simple estimate if key missing
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

    system_prompt = """
You are a certified Ontario (Canada) auto-body damage estimator in the year 2025
with 15+ years of experience. You estimate collision and cosmetic repairs
for retail customers (no deep insurance discounts).

You are given multiple photos of vehicle damage, and possibly a VIN.

Follow this reasoning process INTERNALLY, then output ONLY JSON.

STEP 1: Identify damaged panels
Choose from:
- front bumper upper
- front bumper lower
- rear bumper upper
- rear bumper lower
- left fender
- right fender
- left front door
- right front door
- left rear door
- right rear door
- hood
- trunk
- left quarter panel
- right quarter panel
- rocker panel
- grille area
- headlight area
- taillight area

Be specific. NEVER say "general damage" or "unspecified".

STEP 2: Identify damage types
Choose all that apply:
- dent
- crease dent
- sharp dent
- paint scratch
- deep scratch
- paint scuff
- paint transfer
- crack
- plastic tear
- bumper deformation
- metal distortion
- misalignment
- rust exposure

STEP 3: Suggest repair methods
Choose from:
- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- blend adjacent panels
- recalibration (sensors/cameras)
- refinish only (no structural repair)

STEP 4: Ontario 2025 pricing calibration (CAD)
Use typical Ontario retail pricing:

- PDR: 150–600
- Panel repaint: 350–900
- Panel repair + repaint: 600–1600
- Bumper repaint: 400–900
- Bumper repair + repaint: 750–1400
- Bumper replacement: 800–2000
- Door replacement: 800–2200
- Quarter panel repair: 900–2500
- Quarter panel replacement: 1800–4800
- Hood repaint: 400–900
- Hood replacement: 600–2200

Rules:
- Minor damage → low end
- Moderate → mid range
- Severe or multiple panels → high end or sum across panels
- If multiple panels clearly damaged, sum realistic operations
- If VIN suggests luxury/EV/aluminum, bias 15–30% higher

STEP 5: VIN usage
If a VIN is provided:
- Infer rough segment (economy / mid-range / luxury / truck / EV)
- Adjust cost band appropriately

STEP 6: Output JSON ONLY
Return strictly this JSON (no extra text):

{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [ "front bumper lower", "right fender", ... ],
  "damage_types": [ "dent", "paint scuff", ... ],
  "recommended_repairs": [ "bumper repair + paint", "panel repair + paint", ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean
}
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    content: List[dict] = []
    main_text = "Analyze all uploaded vehicle damage photos and follow the instructions."
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
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
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        result = json.loads(raw)

        # Defaults + sanity
        result.setdefault("severity", "Moderate")
        result.setdefault("damage_areas", [])
        result.setdefault("damage_types", [])
        result.setdefault("recommended_repairs", [])
        result.setdefault("min_cost", 600)
        result.setdefault("max_cost", 1500)
        result.setdefault("confidence", 0.7)
        result.setdefault("vin_used", bool(vin))

        try:
            min_c = float(result["min_cost"])
            max_c = float(result["max_cost"])
            if max_c < min_c:
                min_c, max_c = max_c, min_c
            if max_c - min_c > 6000:
                mid = (min_c + max_c) / 2
                min_c = mid - 1500
                max_c = mid + 1500
            result["min_cost"] = max(100.0, round(min_c))
            result["max_cost"] = max(result["min_cost"] + 50.0, round(max_c))
        except Exception:
            result["min_cost"] = 600
            result["max_cost"] = 1500

        return result

    except Exception as e:
        print("AI estimator error:", e)
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
# HELPERS: SAVE ESTIMATE + ADMIN AUTH
# ============================================================


def save_estimate_to_db(
    shop: ShopConfig,
    phone: str,
    vin: Optional[str],
    result: dict,
) -> str:
    db = SessionLocal()
    try:
        est = Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=result.get("severity"),
            damage_areas=", ".join(result.get("damage_areas", [])),
            damage_types=", ".join(result.get("damage_types", [])),
            recommended_repairs=", ".join(result.get("recommended_repairs", [])),
            min_cost=result.get("min_cost"),
            max_cost=result.get("max_cost"),
            confidence=result.get("confidence"),
            vin=vin,
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


def require_admin(request: Request) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ============================================================
# APPOINTMENT SLOTS
