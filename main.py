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
# TWILIO CLIENT (MEDIA DOWNLOAD)
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
        raise Exception(f"Twilio media download error: {resp.status_code} {resp.text}")

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
    # calendar_id is supported in SHOPS_JSON but not used yet


def load_shops() -> Dict[str, ShopConfig]:
    """
    Load shops from SHOPS_JSON env var.
    Example SHOPS_JSON:
    [
      { "id": "shop1", "name": "Brampton Auto Body", "webhook_token": "brampton123" },
      { "id": "shop2", "name": "Mississauga Collision Centre", "webhook_token": "miss_centre_456" }
    ]
    """
    if not SHOPS_JSON:
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo")
        return {default.webhook_token: default}

    try:
        raw = json.loads(SHOPS_JSON)
        shops: Dict[str, ShopConfig] = {}
        for item in raw:
            # ignore extra keys in JSON (like calendar_id)
            sc = ShopConfig(
                id=item.get("id", "default"),
                name=item.get("name", "Auto Body Shop"),
                webhook_token=item["webhook_token"],
            )
            shops[sc.webhook_token] = sc
        return shops
    except Exception as e:
        print("Error parsing SHOPS_JSON:", e)
        default = ShopConfig(id="default", name="Auto Body Shop", webhook_token="demo")
        return {default.webhook_token: default}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    """Determine shop based on ?token= in webhook URL."""
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
def on_startup():
    Base.metadata.create_all(bind=engine)


# ============================================================
# HELPERS: IMAGES + VIN EXTRACTION
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    """Extract all Twilio MediaUrl0, MediaUrl1, …"""
    urls: List[str] = []
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
# SUPER-PROMPT v3.0 (SMART MODE: BODY + WHEEL DAMAGE)
# ============================================================

AI_SYSTEM_PROMPT = """
You are a certified Ontario (Canada) auto-body and wheel damage estimator
(2025) with 15+ years of experience. You analyze 1–5 PHOTOS of vehicle
damage plus an optional VIN. You produce a realistic RETAIL pre-estimate
for Ontario, Canada (no deep insurance discounts).

You MUST:
- Only use what is clearly visible in the photos + very light SMART inference
  to adjacent panels.
- Never invent panels or damage that are not obviously involved.
- Be conservative and realistic – this is a pre-estimate, not a wild guess.
- Output STRICT JSON ONLY in the defined schema (no extra text).

------------------------------------------------------------
1) IDENTIFY DAMAGED ZONES
------------------------------------------------------------
You may tag:

BODY PANELS:
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
- trunk / liftgate
- left quarter panel
- right quarter panel
- rocker panel / sill
- grille area
- headlight area
- taillight area

WHEELS / RIMS:
- left front wheel / rim
- right front wheel / rim
- left rear wheel / rim
- right rear wheel / rim

RULES:
- If ONLY a wheel is visible with curb rash, do NOT add bumpers or fenders.
- SMART MODE: if the photo clearly shows a bumper AND the adjacent fender
  hit in the same region, you may tag both – but do NOT chain this across
  the car.
- Do not add panels you cannot see or that are not clearly involved.

------------------------------------------------------------
2) DAMAGE TYPES
------------------------------------------------------------
Choose all that apply (per overall case):

- dent
- crease dent
- sharp dent
- paint scratch
- deep scratch
- paint scuff
- paint transfer
- chip / stone chip
- crack
- plastic tear
- bumper deformation
- metal distortion
- misalignment
- rust exposure
- curb rash (for wheels)
- gouge (for wheels or plastic)
- tyre sidewall damage (only if obviously visible)

If unsure between two types, choose the less severe option.

------------------------------------------------------------
3) REPAIR METHODS
------------------------------------------------------------
Examples (choose a realistic subset):

BODY:
- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- blend adjacent panels
- refinish only (no structural repair)
- headlight / taillight replacement
- recalibration (sensors/cameras) – ONLY if front or rear area clearly hit.

WHEELS/TIRES:
- wheel refinish (curb rash repair + repaint)
- wheel replacement
- tyre replacement (ONLY if sidewall damage clearly visible)

------------------------------------------------------------
4) ONTARIO 2025 PRICING (CAD)
------------------------------------------------------------
Use realistic Ontario retail ranges:

GENERAL BODY (per panel/operation):
- PDR: 150–600
- Panel repaint: 350–900
- Panel repair + repaint: 600–1600
- Bumper repaint: 400–900
- Bumper repair + repaint: 750–1400
- Bumper replacement (painted & installed): 800–2200
- Door replacement & paint: 800–2400
- Quarter panel repair & paint: 900–2600
- Quarter panel replacement & paint: 1800–4800
- Hood repaint: 400–900
- Hood replacement & paint: 600–2400
- Headlight or taillight replacement: 250–900 each installed

WHEELS/TIRES (per wheel/tyre):
- Wheel refinish (curb rash repair + repaint): 180–450
- Wheel replacement (OEM, painted & installed): 450–1600
- Tyre replacement (installed & balanced): 180–450

ADJUSTMENTS:
- Economy car → lean lower end of ranges.
- Luxury / EV / large pickup → lean higher end (15–30% more typical).
- Sum realistic operations when multiple panels or wheels are involved.
- For simple cosmetic damage on a single area, keep the range reasonably narrow.

------------------------------------------------------------
5) VIN USAGE
------------------------------------------------------------
If a VIN is provided, use it mentally to infer vehicle type
(economy / luxury / EV / truck / SUV) and adjust costs accordingly.
Set "vin_used": true if it influenced pricing; otherwise false.

------------------------------------------------------------
6) SEVERITY & CONFIDENCE
------------------------------------------------------------
"severity":
- "Minor"   → cosmetic issues, limited deformation, 1–2 areas.
- "Moderate"→ multiple panels or clear deformation; drivable but needs work.
- "Severe"  → heavy damage, multiple panels, possible structural/safety issues.

"confidence" (0.0–1.0):
- 0.9–1.0: clear, well-lit photos, damage obvious.
- 0.6–0.8: usable but some ambiguity (lighting/angle/partial view).
- 0.3–0.5: poor photos, very hard to see; keep prices conservative.

------------------------------------------------------------
7) CUSTOMER SUMMARY
------------------------------------------------------------
Write "customer_summary" as 1–3 short, friendly sentences explaining:
- What is damaged and how badly.
- The general repair approach (refinish vs repair vs replace).
- That an in-person inspection may adjust the final price.

------------------------------------------------------------
8) FINAL OUTPUT (JSON ONLY)
------------------------------------------------------------
Return ONLY a JSON object with exactly these keys:

{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [ "front bumper lower", "right front wheel / rim", ... ],
  "damage_types": [ "dent", "paint scratch", "curb rash", ... ],
  "recommended_repairs": [ "wheel refinish", "panel repair + paint", ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean,
  "customer_summary": "short friendly explanation"
}

Rules:
- CAD only.
- Ensure max_cost >= min_cost.
- Do NOT add extra keys.
- Do NOT output anything outside of JSON.
""".strip()


# ============================================================
# AI DAMAGE ESTIMATION WITH POST-PROCESSING & FLAGS
# ============================================================

async def estimate_damage_from_images(
    image_urls: List[str],
    vin: Optional[str],
    shop: ShopConfig,
) -> dict:
    """Call OpenAI vision model, then apply strict post-processing & sanity checks."""

    # Fallback if no API key
    if not OPENAI_API_KEY:
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.6,
            "vin_used": False,
            "customer_summary": (
                "Our AI estimator is temporarily unavailable. "
                "A technician will review your photos and confirm a quote."
            ),
            "flags": ["no_openai_key"],
        }

    # Build image content for OpenAI
    content: List[dict] = []
    intro = f"Analyze the visible damage for {shop.name} in Ontario, Canada."
    if vin:
        intro += f" Vehicle VIN: {vin}."
    content.append({"type": "text", "text": intro})

    for url in image_urls[:5]:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img_bytes = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img_bytes)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception as e:
                print("Twilio image download error:", e)
        else:
            content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
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
        result = json.loads(raw_str)
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
            "customer_summary": (
                "We had trouble analyzing the photos automatically. "
                "A technician will provide a manual estimate."
            ),
            "flags": ["ai_error"],
        }

    # ---------- Defaults & safety ----------
    result.setdefault("severity", "Moderate")
    result.setdefault("damage_areas", [])
    result.setdefault("damage_types", [])
    result.setdefault("recommended_repairs", [])
    result.setdefault("min_cost", 600)
    result.setdefault("max_cost", 1500)
    result.setdefault("confidence", 0.6)
    result.setdefault("vin_used", bool(vin))
    result.setdefault(
        "customer_summary",
        "This is an AI-based preliminary estimate. "
        "A technician will confirm the final cost in person.",
    )

    # numeric cost sanity
    try:
        min_c = float(result.get("min_cost", 600))
        max_c = float(result.get("max_cost", 1500))
    except Exception:
        min_c, max_c = 600.0, 1500.0

    if max_c < min_c:
        min_c, max_c = max_c, min_c

    # basic range before rule-based tightening
    min_c = max(50.0, round(min_c))
    max_c = max(min_c + 50.0, round(max_c))

    result["min_cost"] = min_c
    result["max_cost"] = max_c

    damage_areas = result.get("damage_areas", []) or []
    damage_types = result.get("damage_types", []) or []
    severity = (result.get("severity") or "Moderate").lower()

    # ---------- POST-PROCESSING RULES ----------
    flags: List[str] = []

    # helper flags
    def _is_wheel_area(area: str) -> bool:
        a = area.lower()
        return "wheel" in a or "rim" in a

    is_wheel_only = len(damage_areas) > 0 and all(_is_wheel_area(a) for a in damage_areas)
    is_single_panel = len(damage_areas) == 1 and not is_wheel_only
    is_minor = severity == "minor"

    # RULE A — WHEEL-ONLY JOBS
    if is_wheel_only:
        MIN_WHEEL = 150
        MAX_WHEEL_REFINISH = 450
        MAX_WHEEL_REPLACE = 1600

        high_severity_wheel = any(
            t in [ "gouge", "crack", "plastic tear" ] for t in [d.lower() for d in damage_types]
        )

        if high_severity_wheel:
            # replacement-like ranges
            result["min_cost"] = max(MIN_WHEEL, result["min_cost"])
            result["max_cost"] = min(MAX_WHEEL_REPLACE, result["max_cost"])
        else:
            # cosmetic curb rash refinish
            if result["min_cost"] < MIN_WHEEL:
                result["min_cost"] = MIN_WHEEL
            if result["max_cost"] > MAX_WHEEL_REFINISH:
                result["max_cost"] = MAX_WHEEL_REFINISH

        if result["max_cost"] > MAX_WHEEL_REPLACE:
            flags.append("wheel_cost_too_high_capped")
            result["max_cost"] = MAX_WHEEL_REPLACE

    # RULE B — VERY MINOR SINGLE-PANEL DAMAGE (non-wheel)
    if is_single_panel and is_minor:
        SAFE_MIN = 150
        SAFE_MAX = 650
        if result["min_cost"] < SAFE_MIN:
            result["min_cost"] = SAFE_MIN
        if result["max_cost"] > SAFE_MAX:
            flags.append("minor_single_panel_capped")
            result["max_cost"] = SAFE_MAX

    # RULE C — GLOBAL SANITY CAP
    HARD_CAP = 12000.0
    SOFT_CAP = 5000.0

    if severity != "severe":
        if result["max_cost"] > SOFT_CAP:
            flags.append("non_severe_soft_capped")
            result["max_cost"] = SOFT_CAP
    else:
        if result["max_cost"] > HARD_CAP:
            flags.append("severe_hard_capped")
            result["max_cost"] = HARD_CAP

    # Additional consistency checks
    if is_wheel_only and result["max_cost"] > 600:
        flags.append("wheel_cost_review")

    if is_single_panel and result["max_cost"] > 1200:
        flags.append("single_panel_high_cost_review")

    if is_minor and result["max_cost"] > 1500:
        flags.append("minor_cost_inconsistent")

    if severity == "moderate" and result["max_cost"] > 6000:
        flags.append("moderate_cost_excessive")

    # Re-sort final costs after all clamps
    if result["max_cost"] < result["min_cost"]:
        result["max_cost"] = result["min_cost"] + 50

    result["flags"] = flags

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
# APPOINTMENT SLOTS
# ============================================================

def get_appointment_slots(n: int = 3) -> List[datetime.datetime]:
    """Generate up to n appointment slots for tomorrow."""
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]  # 9am, 11am, 2pm, 4pm

    slots: List[datetime.datetime] = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Auto-shop AI estimator running",
        "hint": "Configure Twilio webhook as POST to /sms-webhook?token=YOUR_SHOP_TOKEN",
    }


# ============================================================
# TWILIO SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    """
    Twilio SMS entrypoint:
    - If customer replies 1/2/3 after estimate → book appointment.
    - If customer sends images → run AI estimator, save to DB, offer slots.
    - If no images → send instructions.
    """
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # 1) Booking selection flow (reply 1/2/3)
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots: List[datetime.datetime] = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]
            lines = [
                f"You're booked at {shop.name}.",
                "",
                "Appointment time:",
                chosen.strftime("%a %b %d at %I:%M %p"),
                "",
                "If you need to change this time, reply 'Change'.",
            ]
            reply.message("\n".join(lines))

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # 2) Image-based AI estimate flow
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result.get("severity", "Moderate")
        min_cost = result.get("min_cost", 600)
        max_cost = result.get("max_cost", 1500)
        areas = ", ".join(result.get("damage_areas", [])) or "Not clearly detected"
        types = ", ".join(result.get("damage_types", [])) or "Not clearly identified"
        summary = result.get("customer_summary", "")
        flags = result.get("flags", [])

        if max_cost > 0:
            cost_range = f"${min_cost:,.0f} – ${max_cost:,.0f}"
        else:
            cost_range = "N/A"

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        msg_lines = [
            f"AI Damage Estimate for {shop.name}",
            "",
            f"Severity: {severity}",
            f"Estimated Cost (Ontario 2025): {cost_range}",
            f"Areas: {areas}",
            f"Damage Types: {types}",
        ]

        if summary:
            msg_lines.append("")
            msg_lines.append(summary)

        if flags:
            msg_lines.append("")
            msg_lines.append(f"Internal flags: {', '.join(flags)}")

        msg_lines.append("")
        msg_lines.append(f"Estimate ID (internal): {estimate_id}")
        msg_lines.append("")
        msg_lines.append("Reply with a number to book an in-person estimate:")

        for i, s in enumerate(slots, 1):
            msg_lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(msg_lines))
        return Response(content=str(reply), media_type="application/xml")

    # 3) No images → onboarding message
    intro_lines = [
        f"Thanks for messaging {shop.name}.",
        "",
        "To get an AI-powered pre-estimate:",
        "- Send 1–5 clear photos of the damage",
        "- Optional: include your 17-character VIN in the text",
    ]

    reply.message("\n".join(intro_lines))
    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# ADMIN API (READ-ONLY)
# ============================================================

def require_admin(request: Request):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


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
        rows = q.all()
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
            for e in rows
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
