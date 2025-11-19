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
    We assume JPEG, which is what Twilio usually sends for photos.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# SHOP CONFIG
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    # extra keys (like calendar_id) in SHOPS_JSON will be ignored by Pydantic


def load_shops() -> Dict[str, ShopConfig]:
    """Parse SHOPS_JSON env var into token->ShopConfig map."""
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
    except Exception as e:
        print("Failed to parse SHOPS_JSON:", e)
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        return {default.webhook_token: default}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    """Pick shop based on ?token= in the Twilio webhook URL."""
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
# CONSTANTS FOR POST-PROCESSING
# ============================================================

SEVERITY_LEVELS = ["Minor", "Moderate", "Severe"]

WHEEL_KEYWORDS = ("wheel", "rim", "tyre", "tire")
GLASS_KEYWORDS = ("windshield", "rear glass", "window", "glass", "sunroof")

WHEEL_DAMAGE_TYPES = {"paint scuff", "paint scratch", "curb rash", "chip", "gouge"}
STRUCTURAL_DAMAGE_TYPES = {
    "plastic tear",
    "metal distortion",
    "bumper deformation",
    "frame damage",
    "glass crack",
    "glass shatter",
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def is_wheel_only(areas: List[str]) -> bool:
    if not areas:
        return False
    areas_l = [a.lower() for a in areas]
    return all(any(k in a for k in WHEEL_KEYWORDS) for a in areas_l)


def postprocess_damage_result(raw: dict) -> dict:
    """
    Tighten the AI output so results are realistic for Ontario 2025
    body-shops, with special handling for wheel-only, light damage, etc.
    """
    result = dict(raw)  # shallow copy so we can modify safely

    # 1) Prefer visible_areas over generic damage_areas
    damage_areas = result.get("visible_areas") or result.get("damage_areas") or []
    if not isinstance(damage_areas, list):
        damage_areas = []
    damage_areas = [a.strip() for a in damage_areas if isinstance(a, str) and a.strip()]
    result["damage_areas"] = damage_areas

    # 2) Normalize damage_types
    damage_types = result.get("damage_types") or []
    if not isinstance(damage_types, list):
        damage_types = []
    damage_types = [t.strip() for t in damage_types if isinstance(t, str) and t.strip()]
    result["damage_types"] = damage_types

    # 3) Normalize recommended_repairs
    rec_repairs = result.get("recommended_repairs") or []
    if not isinstance(rec_repairs, list):
        rec_repairs = []
    rec_repairs = [r.strip() for r in rec_repairs if isinstance(r, str) and r.strip()]
    result["recommended_repairs"] = rec_repairs

    # 4) Normalize severity
    severity = result.get("severity") or "Moderate"
    if severity not in SEVERITY_LEVELS:
        # Try case-insensitive match
        sev_lower = severity.lower()
        if "minor" in sev_lower:
            severity = "Minor"
        elif "severe" in sev_lower:
            severity = "Severe"
        else:
            severity = "Moderate"
    result["severity"] = severity

    # 5) Normalize costs
    try:
        min_cost = float(result.get("min_cost", 600.0))
        max_cost = float(result.get("max_cost", 1500.0))
    except Exception:
        min_cost, max_cost = 600.0, 1500.0

    if max_cost < min_cost:
        min_cost, max_cost = max_cost, min_cost

    # Hard sanity: estimates shouldn't be insane for SMS pre-quotes
    min_cost = clamp(min_cost, 50.0, 15000.0)
    max_cost = clamp(max_cost, min_cost + 50.0, 20000.0)

    # 6) Special: wheel-only damage sanity
    if is_wheel_only(damage_areas):
        # Pure cosmetic wheel/tire damage should not be a $5k+ severe estimate
        cosmetic = all(
            (t.lower() in WHEEL_DAMAGE_TYPES) or ("scuff" in t.lower())
            for t in damage_types
        ) or not damage_types

        if cosmetic:
            severity = "Minor"
            # Typical Ontario retail: wheel refinish/repair per wheel
            # We'll assume 1–2 wheels from a single photo
            min_cost = 180.0
            max_cost = 850.0

        result["severity"] = severity
        result["min_cost"] = round(min_cost)
        result["max_cost"] = round(max_cost)
        # Confidence can stay, but if missing, set mid
        result["confidence"] = float(result.get("confidence", 0.75))
        return result

    # 7) General logic: if really low money, don't call it Severe
    if max_cost <= 1000.0 and severity == "Severe":
        severity = "Moderate"
    if max_cost <= 600.0 and severity == "Moderate":
        severity = "Minor"

    # 8) If very high $$ and obvious structural damage, bump severity up
    if max_cost >= 4000.0 and any(
        s in {t.lower() for t in damage_types} for s in STRUCTURAL_DAMAGE_TYPES
    ):
        severity = "Severe"

    result["severity"] = severity

    # 9) Clamp band width to something reasonable (no $0–$20k swings)
    if max_cost - min_cost > 8000.0:
        mid = (min_cost + max_cost) / 2.0
        min_cost = max(200.0, mid - 2500.0)
        max_cost = mid + 2500.0

    result["min_cost"] = round(min_cost)
    result["max_cost"] = round(max_cost)

    # 10) Confidence default
    try:
        conf = float(result.get("confidence", 0.75))
    except Exception:
        conf = 0.75
    result["confidence"] = clamp(conf, 0.2, 0.99)

    # 11) Ensure vin_used is boolean
    result["vin_used"] = bool(result.get("vin_used", False))

    # 12) Customer summary fallback
    if not result.get("customer_summary"):
        result["customer_summary"] = (
            "This is an AI-based pre-estimate. A technician will confirm final "
            "pricing after an in-person inspection."
        )

    return result


# ============================================================
# AI DAMAGE ESTIMATION (ONTARIO 2025, MULTI-IMAGE + TWILIO MEDIA)
# ============================================================

async def estimate_damage_from_images(
    image_urls: List[str],
    vin: Optional[str],
    shop: ShopConfig,
) -> dict:
    """
    Call OpenAI vision model, or return a safe fallback if key missing.
    Supports Twilio private media by downloading and converting to
    base64 data URLs.
    """
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
            "customer_summary": (
                "Based on the photos, we estimate moderate cosmetic damage. "
                "A detailed in-person inspection may adjust the final cost."
            ),
        }

    system_prompt = '''
You are a certified Ontario (Canada) auto-body damage estimator in the year 2025
with 15+ years of experience. You estimate collision and cosmetic repairs
for retail customers (no deep insurance discounts).

You are given multiple PHOTOS of vehicle damage, and sometimes a VIN.
Your job is to produce a conservative but realistic PRE-ESTIMATE, not a full DRP sheet.

CRITICAL RULES ABOUT WHAT YOU CAN SAY
------------------------------------
1) ONLY describe panels and parts that are clearly visible in the photos.
   - If you cannot clearly see a part, DO NOT list it in "visible_areas".
   - Do NOT assume front damage from a rear photo or vice versa.
   - Do NOT assume the opposite side is damaged if you cannot see it.

2) If the photo shows ONLY a wheel/rim/tire:
   - Limit "visible_areas" to that wheel/rim/tire.
   - Do NOT add bumpers, fenders, doors, or lights unless they are clearly damaged in the photo.

3) If you are uncertain:
   - Use "possible_hidden_areas" for educated guesses,
     but keep "visible_areas" STRICTLY to what you clearly see.

4) This is for pro body shops. They care that:
   - Damage labels (panels/areas) match what’s actually in the picture.
   - Cost bands are realistic for Ontario 2025 retail rates.

PANEL / AREA VOCABULARY
-----------------------
When listing areas, only use items from this controlled vocabulary
and only if they are ACTUALLY VISIBLE in the photos:

FRONT:
- front bumper upper
- front bumper lower
- grille area
- hood
- left front fender
- right front fender
- left headlight
- right headlight
- front left wheel / rim
- front right wheel / rim
- front left tire
- front right tire
- left mirror
- right mirror
- front windshield

SIDE / BODY:
- left front door
- right front door
- left rear door
- right rear door
- left rocker panel
- right rocker panel
- left quarter panel
- right quarter panel
- roof
- left front window
- right front window
- left rear window
- right rear window

REAR:
- rear bumper upper
- rear bumper lower
- trunk
- rear glass
- left taillight
- right taillight
- rear left wheel / rim
- rear right wheel / rim
- rear left tire
- rear right tire

WHEEL / TIRE ONLY:
- front left wheel / rim
- front right wheel / rim
- rear left wheel / rim
- rear right wheel / rim
- front left tire
- front right tire
- rear left tire
- rear right tire

DAMAGE TYPE VOCABULARY
----------------------
Choose all that apply, only if you can really see them:

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
- curb rash
- glass chip
- glass crack
- glass shatter

REPAIR METHOD VOCABULARY
------------------------
Choose all that apply:

- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- blend adjacent panels
- refinish wheel
- replace wheel
- tire replacement
- glass chip repair
- glass replacement
- recalibration (sensors/cameras)
- refinish only (no structural repair)

ONTARIO 2025 PRICING GUIDANCE (CAD, RETAIL)
-------------------------------------------
Use these as reference ranges for pre-estimates:

- PDR (small dent):          150–600
- Panel repaint only:        350–900
- Panel repair + repaint:    600–1600
- Bumper repaint:            400–900
- Bumper repair + paint:     750–1400
- Bumper replacement:        800–2200
- Door replacement:          800–2200
- Quarter panel repair:      900–2500
- Quarter panel replacement: 1800–4800
- Hood repaint:              400–900
- Hood replacement:          600–2200
- Wheel refinish:            180–450 per wheel
- Wheel replacement:         450–1200 per wheel
- Windshield replacement:    450–1200
- Glass chip repair:         80–250

Rules:
- Minor, isolated cosmetic damage → low end
- Moderate multi-panel damage → mid range
- Heavy structural / multiple panels / glass + panels → high end
- If VIN indicates luxury/EV/aluminum, it’s okay to bias 15–30% higher,
  but still stay realistic for a pre-estimate.

VIN USAGE
---------
If a VIN is provided:
- Infer vehicle segment (economy / mid-range / luxury / truck / EV).
- Adjust cost band appropriately.
- Set "vin_used" to true only if VIN actually influenced your pricing.

OUTPUT FORMAT (VERY IMPORTANT)
------------------------------
You MUST return VALID JSON ONLY, with EXACTLY these keys:

{
  "severity": "Minor" | "Moderate" | "Severe",
  "visible_areas": [ "rear bumper lower", "left quarter panel", ... ],
  "possible_hidden_areas": [ "rear body structure behind bumper", ... ],
  "damage_types": [ "dent", "paint scratch", ... ],
  "recommended_repairs": [ "bumper repair + paint", "refinish wheel", ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean,
  "customer_summary": "1–3 sentence explanation in friendly language",
  "reasoning_notes": "Very short internal notes about why you chose panels, severity, and cost."
}

STRICT VISIBILITY RULE:
- "visible_areas" must ONLY contain panels you can literally see and identify in the photos.
- Use "possible_hidden_areas" for anything that is a guess or might be behind trim.
- NEVER invent front-end damage from a rear-only photo, or vice versa.
- NEVER add more wheels/tires than are clearly visible.

Think slowly, like a professional estimator, then output ONLY the JSON.
'''.strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build OpenAI content payload
    content: List[dict] = []
    main_text = (
        f"Analyze these vehicle damage photos for {shop.name} "
        "and follow ALL instructions in the system prompt very carefully."
    )
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    # Use at most 3 images per request to keep latency and cost reasonable
    usable_urls = image_urls[:3]

    for url in usable_urls:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img_bytes = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img_bytes)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception as e:
                print("Error downloading Twilio media:", e)
        else:
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
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        raw_result = json.loads(raw_content)

        # Run our Python-side sanity pass
        result = postprocess_damage_result(raw_result)
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
            "customer_summary": (
                "We had trouble analyzing the photos, but this looks like "
                "moderate damage. A technician will confirm final pricing "
                "after an in-person inspection."
            ),
        }


# ============================================================
# HELPERS: SAVE ESTIMATE + ADMIN AUTH
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


def require_admin(request: Request):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ============================================================
# APPOINTMENT SLOTS
# ============================================================

def get_appointment_slots(n: int = 3) -> List[datetime.datetime]:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    slots: List[datetime.datetime] = []
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
    """
    Main Twilio SMS entrypoint.
    - If 1–5 images: run AI estimator, save to DB, return estimate + time slots.
    - If no images: send instructions.
    - If user replies 1/2/3 after estimate: book a time (session-based).
    """
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # 1) Booking selection flow
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

    # 2) Multi-image AI estimate (if images present)
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        min_cost = result["min_cost"]
        max_cost = result["max_cost"]
        cost_range = f"${min_cost:,.0f} – ${max_cost:,.0f}"

        areas = ", ".join(result.get("damage_areas", [])) or "visible panels identified"
        types = ", ".join(result.get("damage_types", [])) or "detailed damage types identified"
        summary = result.get("customer_summary") or ""

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"AI Damage Estimate for {shop.name}",
            "",
            f"Severity: {severity}",
            f"Estimated Cost (Ontario 2025): {cost_range}",
            f"Areas: {areas}",
            f"Damage Types: {types}",
        ]

        if summary:
            lines.append("")
            lines.append(summary)

        lines.append("")
        lines.append(f"Estimate ID (internal): {estimate_id}")
        lines.append("")
        lines.append("Reply with a number to book an in-person estimate:")

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # 3) Default onboarding message (no images)
    intro_lines = [
        f"Thanks for messaging {shop.name}.",
        "",
        "To get an AI-powered pre-estimate:",
        "- Send 1–5 clear photos of the damage",
        "- Try to show the whole damaged area and nearby panels",
        "- Optional: include your 17-character VIN in the text",
    ]

    reply.message("\n".join(intro_lines))
    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# SIMPLE ADMIN API (READ-ONLY)
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
