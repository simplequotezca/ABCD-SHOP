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
# ENVIRONMENT + DATABASE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Attach Postgres in Railway and set DATABASE_URL."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

app = FastAPI()

# ============================================================
# TWILIO MEDIA HELPERS
# ============================================================

twilio_client: Optional[Client] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def download_twilio_image(media_url: str) -> bytes:
    """Download image from Twilio MMS securely."""
    if not twilio_client:
        raise RuntimeError("Twilio not configured")

    resp = twilio_client.request("GET", media_url)
    if resp.status_code != 200:
        raise Exception(f"Twilio media download error: {resp.status_code}")

    return resp.content


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 image URL."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# SHOP CONFIGURATION (MULTI-SHOP)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """Load shops from SHOPS_JSON env var."""
    if not SHOPS_JSON:
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo")
        return {default.webhook_token: default}

    try:
        arr = json.loads(SHOPS_JSON)
        shops = {}
        for s in arr:
            sc = ShopConfig(**s)
            shops[sc.webhook_token] = sc
        return shops
    except:
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo")
        return {default.webhook_token: default}


SHOPS_BY_TOKEN = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    """Determine shop based on ?token= in webhook."""
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
    customer_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    # ============================================================
# HELPERS: IMAGES + VIN EXTRACTION
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    """Extract all MediaUrl0, MediaUrl1, … from Twilio MMS."""
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
    """Extract VIN if present in message body."""
    if not text:
        return None
    match = VIN_PATTERN.search(text.upper())
    return match.group(1) if match else None


# ============================================================
# AI DAMAGE ESTIMATION - ULTRA ACCURATE SYSTEM PROMPT
# ============================================================

AI_SYSTEM_PROMPT = """
You are a certified Ontario (Canada) auto-body estimator (2025) with 15+ years expertise.
You analyze vehicle damage from photos and output strict JSON only.

Follow these steps INTERNALLY before answering:

1. Identify all damaged panels only from:
   - front bumper upper/lower
   - rear bumper upper/lower
   - left/right fender
   - left/right front door
   - left/right rear door
   - hood, trunk
   - left/right quarter panel
   - rocker panel
   - grille area
   - headlight/taillight area

2. Identify all damage types:
   dent, crease dent, sharp dent, paint scratch, deep scratch,
   paint scuff, paint transfer, crack, plastic tear, bumper deformation,
   metal distortion, misalignment, rust exposure

3. Recommend repairs:
   PDR, panel repair + paint, bumper repair + paint, bumper replacement,
   panel replacement, blend adjacent panels, recalibration, refinish only

4. Ontario 2025 pricing (CAD):
   PDR 150–600
   Panel repaint 350–900
   Panel repair+paint 600–1600
   Bumper repaint 400–900
   Bumper repair+paint 750–1400
   Bumper replacement 800–2000
   Door replacement 800–2200
   Quarter panel repair 900–2500
   Quarter panel replacement 1800–4800
   Hood repaint 400–900
   Hood replacement 600–2200

   - Minor damage → low
   - Moderate → mid
   - Severe or multi-panel → high or sum
   - Luxury/EV (VIN) → +15–30%

5. Output JSON EXACTLY:
{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [...],
  "damage_types": [...],
  "recommended_repairs": [...],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean,
  "customer_summary": "1–2 friendly sentences"
}
""".strip()
async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig):
    if not OPENAI_API_KEY:
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.65,
            "vin_used": False,
            "customer_summary": "Moderate cosmetic damage suggested. Technician will confirm in person."
        }

    # Build content
    content = []
    header_text = f"Analyze damage for {shop.name}." + (f" VIN: {vin}" if vin else "")
    content.append({"type": "text", "text": header_text})

    for url in image_urls[:2]:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception as e:
                print("Twilio image error:", e)
        else:
            content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        result = json.loads(raw)
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
            "customer_summary": "Preliminary estimate suggests moderate damage. Technician will confirm."
        }

    # Safety corrections
    try:
        lo = float(result.get("min_cost", 600))
        hi = float(result.get("max_cost", 1500))
        if hi < lo:
            lo, hi = hi, lo
        result["min_cost"] = max(100, round(lo))
        result["max_cost"] = max(result["min_cost"] + 50, round(hi))
    except:
        result["min_cost"] = 600
        result["max_cost"] = 1500

    result.setdefault("vin_used", bool(vin))
    return result
    # ============================================================
# SAVE ESTIMATE TO DATABASE
# ============================================================

def save_estimate_to_db(shop: ShopConfig, phone: str, vin: Optional[str], result: dict) -> str:
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
            customer_summary=result.get("customer_summary"),
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


# ============================================================
# APPOINTMENT TIMES
# ============================================================

def get_appointment_slots():
    """Generate next-day appointment slots at 9am, 11am, 2pm, 4pm."""
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    return [
        tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in hours
    ]


# ============================================================
# TWILIO WEBHOOK ROUTE
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    """
    Main SMS webhook.
    1) User replies 1/2/3 → Book appointment
    2) User sends images → Generate AI estimate
    3) User sends no images → Send instructions
    """
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    # User session tracking
    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # ========================================================
    # 1) BOOKING FLOW — USER REPLIES WITH 1,2,3
    # ========================================================
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]
            reply.message(
                f"You're booked at {shop.name}!\n\n"
                f"Appointment time:\n{chosen.strftime('%a %b %d at %I:%M %p')}\n\n"
                f"Reply 'Change' to reschedule."
            )

            session["awaiting_time"] = False
            SESSIONS[session_key] = session
            return Response(content=str(reply), media_type="application/xml")

    # ========================================================
    # 2) DAMAGE ESTIMATION — IF IMAGES ARE PROVIDED
    # ========================================================
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        est_id = save_estimate_to_db(shop, from_number, vin, result)

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        msg = [
            f"AI Damage Estimate for {shop.name}",
            "",
            f"Severity: {result['severity']}",
            f"Estimated Cost (Ontario 2025): ${result['min_cost']} – ${result['max_cost']}",
            f"Panels: {', '.join(result['damage_areas']) or 'Not detected'}",
            f"Damage Types: {', '.join(result['damage_types']) or 'Not detected'}",
            "",
            result.get("customer_summary", ""),
            "",
            f"Estimate ID: {est_id}",
            "",
            "Choose a time for an in-person estimate:",
        ]

        for i, s in enumerate(slots, 1):
            msg.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(msg))
        return Response(content=str(reply), media_type="application/xml")

    # ========================================================
    # 3) NO IMAGES — SEND INSTRUCTIONS
    # ========================================================
    reply.message(
        f"Thanks for messaging {shop.name}.\n\n"
        "To get an AI-generated pre-estimate:\n"
        "- Send 1–5 clear photos of the damage\n"
        "- Optional: include your 17-digit VIN"
    )
    return Response(content=str(reply), media_type="application/xml")
    # ============================================================
# ADMIN AUTH
# ============================================================

def require_admin(request: Request):
    """Simple x-api-key header or ?api_key= check."""
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ============================================================
# ADMIN: LIST ALL ESTIMATES
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


# ============================================================
# ADMIN: GET A SPECIFIC ESTIMATE
# ============================================================

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
