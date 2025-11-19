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
# SHOP CONFIG (MULTI-SHOP SUPPORT)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None
    # You can optionally add more per-shop fields later
    # e.g. default region, labour rate, etc.


def load_shops() -> Dict[str, ShopConfig]:
    """
    Parse SHOPS_JSON into token->ShopConfig map.

    SHOPS_JSON example:

    [
      {
        "id": "shop1",
        "name": "Brampton Auto Body",
        "webhook_token": "brampton123",
        "calendar_id": null
      },
      {
        "id": "shop2",
        "name": "Mississauga Collision Centre",
        "webhook_token": "miss_centre_456",
        "calendar_id": null
      }
    ]
    """
    if not SHOPS_JSON:
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        return {default.webhook_token: default}

    try:
        data = json.loads(SHOPS_JSON)
    except Exception as e:
        print("Failed to parse SHOPS_JSON:", e)
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        return {default.webhook_token: default}

    shops: Dict[str, ShopConfig] = {}
    for s in data:
        try:
            shop = ShopConfig(**s)
            shops[shop.webhook_token] = shop
        except Exception as e:
            print("Invalid shop entry in SHOPS_JSON:", s, e)
    if not shops:
        default = ShopConfig(
            id="default", name="Auto Body Shop", webhook_token="demo_token"
        )
        shops[default.webhook_token] = default
    return shops


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
# PANEL + DAMAGE CONSTANTS (for post-processing)
# ============================================================

CANONICAL_AREAS = [
    # front
    "front bumper upper",
    "front bumper lower",
    "hood",
    "grille area",
    "headlight area",
    "left fender",
    "right fender",
    # sides
    "left front door",
    "right front door",
    "left rear door",
    "right rear door",
    "left quarter panel",
    "right quarter panel",
    "rocker panel",
    "mirror left",
    "mirror right",
    "left side glass",
    "right side glass",
    # rear
    "rear bumper upper",
    "rear bumper lower",
    "trunk",
    "taillight area",
    "rear glass",
    # roof / glass
    "roof",
    "windshield",
    # wheels & tires
    "front left wheel / rim",
    "front right wheel / rim",
    "rear left wheel / rim",
    "rear right wheel / rim",
    "front left tire",
    "front right tire",
    "rear left tire",
    "rear right tire",
]

FRONT_HINTS = {
    "front bumper upper",
    "front bumper lower",
    "hood",
    "grille area",
    "headlight area",
}

REAR_HINTS = {
    "rear bumper upper",
    "rear bumper lower",
    "trunk",
    "taillight area",
    "rear glass",
}

GLASS_AREAS = {
    "windshield",
    "rear glass",
    "left side glass",
    "right side glass",
    "headlight area",
    "taillight area",
}

WHEEL_AREAS = {
    "front left wheel / rim",
    "front right wheel / rim",
    "rear left wheel / rim",
    "rear right wheel / rim",
}

TIRE_AREAS = {
    "front left tire",
    "front right tire",
    "rear left tire",
    "rear right tire",
}

SEVERITIES = {"Minor", "Moderate", "Severe"}


def normalize_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v).strip() for v in value if str(v).strip()]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# POST-PROCESSING / SANITY CHECKS
# ============================================================

def postprocess_estimate(raw: dict, vin: Optional[str]) -> dict:
    """
    Clean + stabilize model output so shops get realistic numbers.
    """
    result = dict(raw) if raw else {}

    # Ensure basic fields exist
    result.setdefault("severity", "Moderate")
    result.setdefault("damage_areas", [])
    result.setdefault("damage_types", [])
    result.setdefault("recommended_repairs", [])
    result.setdefault("min_cost", 600)
    result.setdefault("max_cost", 1500)
    result.setdefault("confidence", 0.7)
    result.setdefault("vin_used", bool(vin))
    result.setdefault(
        "customer_summary",
        "Based on the photos, we estimate collision damage that will require professional repair. "
        "A detailed in-person inspection may adjust the final cost.",
    )

    # Normalize severity
    sev = str(result.get("severity", "Moderate")).strip().title()
    if sev not in SEVERITIES:
        # simple mapping
        if "minor" in sev.lower():
            sev = "Minor"
        elif "severe" in sev.lower() or "major" in sev.lower():
            sev = "Severe"
        else:
            sev = "Moderate"
    result["severity"] = sev

    # Normalize arrays
    areas = normalize_list(result.get("damage_areas"))
    types = normalize_list(result.get("damage_types"))
    repairs = normalize_list(result.get("recommended_repairs"))

    # Lowercase helpers
    areas_l = [a.lower() for a in areas]
    types_l = [t.lower() for t in types]

    # --- FRONT vs REAR sanity: if clear rear hints but model said front, flip ---
    has_front_hint = any(a in FRONT_HINTS for a in areas)
    has_rear_hint = any(a in REAR_HINTS for a in areas) or any("taillight" in a for a in areas_l)

    def flip_front_to_rear(a: str) -> str:
        al = a.lower()
        if "front bumper upper" in al:
            return "rear bumper upper"
        if "front bumper lower" in al:
            return "rear bumper lower"
        if "headlight" in al:
            return "taillight area"
        if "hood" in al:
            return "trunk"
        if "grille" in al:
            return "taillight area"
        return a

    # If we clearly have rear-only context, but front bumper is present and no headlight:
    if has_rear_hint and not has_front_hint:
        new_areas: List[str] = []
        for a in areas:
            new_areas.append(flip_front_to_rear(a))
        areas = new_areas
        areas_l = [a.lower() for a in areas]

    # --- Glass logic ---
    has_glass_damage = any(
        (ga in GLASS_AREAS)
        or ("windshield" in ga)
        or ("glass" in ga)
        for ga in areas_l
    ) or any("glass" in t or "windshield" in t for t in types_l)

    # --- Wheel / tire logic ---
    has_wheel_damage = any(ga in WHEEL_AREAS for ga in areas) or any(
        "wheel" in ga or "rim" in ga for ga in areas_l
    ) or any("curb rash" in t or "rim" in t or "wheel" in t for t in types_l)

    has_tire_damage = any(ga in TIRE_AREAS for ga in areas) or any(
        "tire" in ga for ga in areas_l
    ) or any("tire" in t for t in types_l)

    # --- Structural-ish hints ---
    structural_keywords = ["frame", "rail", "buckled", "kink", "airbag", "deployed"]
    has_structural_hint = any(sk in " ".join(types_l) for sk in structural_keywords)

    # --- Cost sanity ---
    try:
        min_c = float(result.get("min_cost", 600))
        max_c = float(result.get("max_cost", 1500))
    except Exception:
        min_c, max_c = 600.0, 1500.0

    if max_c < min_c:
        min_c, max_c = max_c, min_c

    # Base clamps by severity
    if sev == "Minor":
        min_c = clamp(min_c, 50, 1500)
        max_c = clamp(max_c, min_c + 50, 3000)
    elif sev == "Moderate":
        min_c = clamp(min_c, 400, 2500)
        max_c = clamp(max_c, min_c + 200, 7000)
    else:  # Severe
        min_c = clamp(min_c, 800, 4000)
        max_c = clamp(max_c, min_c + 300, 12000)

    # Glass must not be super cheap
    if has_glass_damage:
        min_c = max(min_c, 350)
        max_c = max(max_c, min_c + 150)
        if sev == "Minor":
            sev = "Moderate"

    # Wheel-only (curb rash, no big collision): keep in reasonable band
    if has_wheel_damage and not has_structural_hint and len(areas) <= 3 and not has_glass_damage:
        # single wheel refinishing range
        min_c = clamp(min_c, 150, 500)
        max_c = clamp(max_c, min_c + 100, 900)
        if sev == "Severe":
            sev = "Moderate"

    # Tire sidewall or clear mechanical: bump severity at least Moderate
    if has_tire_damage:
        min_c = max(min_c, 250)
        max_c = max(max_c, min_c + 150)
        if sev == "Minor":
            sev = "Moderate"

    # Too many panels → Severe
    if len(areas) >= 4:
        sev = "Severe"
        min_c = max(min_c, 1200)
        max_c = max(max_c, min_c + 400)

    # Structural hints → Severe & higher band
    if has_structural_hint:
        sev = "Severe"
        min_c = max(min_c, 1800)
        max_c = max(max_c, min_c + 600)

    # Hard sanity: Minor but very high cost makes no sense
    if sev == "Minor" and max_c > 3500:
        sev = "Moderate"

    # Final rounding
    min_c = max(50.0, round(min_c))
    max_c = max(min_c + 50.0, round(max_c))

    result["severity"] = sev
    result["damage_areas"] = areas
    result["damage_types"] = types
    result["recommended_repairs"] = repairs
    result["min_cost"] = min_c
    result["max_cost"] = max_c

    # Confidence safety
    try:
        conf = float(result.get("confidence", 0.7))
    except Exception:
        conf = 0.7
    result["confidence"] = clamp(conf, 0.2, 0.99)

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
    Call OpenAI vision model, then pass through post-processing.
    Supports Twilio private media by downloading and converting to
    base64 data URLs.
    """
    if not OPENAI_API_KEY:
        # Very safe fallback if key missing
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

    system_prompt = """
You are a certified Ontario (Canada) auto-body damage estimator in the year 2025
with 15+ years of experience. You estimate collision and cosmetic repairs
for retail customers (no deep insurance discounts).

You are given multiple photos of vehicle damage, and possibly a VIN.
Assume photos may show ANY exterior part of a car, SUV, van, or pickup.

IMPORTANT ORIENTATION RULES
- "Front" means the end with headlights, grille, and front bumper.
- "Rear" means the end with taillights, trunk or hatch, and rear bumper.
- "Left" and "right" are from the driver's seat facing forward.
- NEVER confuse front vs rear. If you clearly see taillights or trunk,
  you are at the rear. If you clearly see headlights or grille, you are at the front.

STEP 1: Identify damaged panels / areas (BE SPECIFIC)
Choose only from this controlled list where it fits, reusing exact phrases:

FRONT:
- front bumper upper
- front bumper lower
- hood
- grille area
- headlight area
- left fender
- right fender

SIDES:
- left front door
- right front door
- left rear door
- right rear door
- left quarter panel
- right quarter panel
- rocker panel
- mirror left
- mirror right
- left side glass
- right side glass

REAR:
- rear bumper upper
- rear bumper lower
- trunk
- taillight area
- rear glass

ROOF / GLASS:
- roof
- windshield

WHEELS / TIRES:
- front left wheel / rim
- front right wheel / rim
- rear left wheel / rim
- rear right wheel / rim
- front left tire
- front right tire
- rear left tire
- rear right tire

If you see damage that doesn't match any of these (for example, a step bar on a truck),
still choose the closest area (e.g. rocker panel for side step).

STEP 2: Identify damage types (controlled vocabulary)
Use ONLY these phrases where they fit:

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
- panel gap irregular
- glass crack
- glass shatter
- glass chip
- curb rash
- wheel gouge
- wheel bend
- tire sidewall damage
- rust exposure

STEP 3: Suggest repair methods
Use ONLY these where appropriate:

- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- wheel refinishing
- wheel replacement
- tire replacement
- glass repair
- glass replacement
- headlight replacement
- taillight replacement
- blend adjacent panels
- recalibration (sensors/cameras)
- refinish only (no structural repair)
- structural inspection

STEP 4: Ontario 2025 pricing calibration (CAD)
Assume typical Ontario RETAIL body shop pricing (labour + materials + paint).
Use realistic ranges:

- PDR: 150–600 per panel
- Panel repaint: 350–900
- Panel repair + repaint: 600–1600
- Bumper repaint: 400–900
- Bumper repair + repaint: 750–1600
- Bumper replacement: 900–2500
- Door replacement: 900–2600
- Fender repair + paint: 600–1800
- Quarter panel repair: 900–2500
- Quarter panel replacement: 2000–4800
- Hood repaint: 400–900
- Hood replacement: 700–2200
- Roof repair + paint: 700–2200
- Windshield replacement: 350–900
- Rear glass replacement: 400–1000
- Side glass replacement: 250–800
- Headlight replacement (OEM): 450–1600
- Taillight replacement (OEM): 300–1200
- Wheel refinishing (curb rash): 150–500 per wheel
- Wheel replacement (damaged/bent): 400–1900
- Tire replacement (sidewall damage): 200–600
- ADAS / sensor recalibration: 150–600

CALCULATION GUIDELINES:
- MINOR: small scuffs, light dents, one panel, no glass broken, wheel not bent.
  => lower end of ranges.
- MODERATE: multiple panels, bumper corners, some deformation, no frame bending.
  => mid-range totals.
- SEVERE: heavy collision, smashed lamps, major deformation, wheel pushed back,
  obvious structural concern.
  => high-end or combined costs following realistic body shop logic.

If multiple panels or operations are clearly required, ADD them together realistically.
Do NOT output extreme ranges like $0–$10,000. Stay within the bands above.

STEP 5: VIN usage (if provided)
If a VIN is included:
- Infer rough class (economy / mid-range / luxury / truck / EV) from make and model.
- Luxury / EV / full-size trucks usually cost 15–30% more.
- Bias your min and max costs accordingly but still stay realistic.

STEP 6: Choose severity
- "Minor" = cosmetic only, no broken lights/glass, no obvious structure, no wheel/tire issues.
- "Moderate" = several panels, bumper plastics cracked or deformed, lamps possibly broken,
  but no obvious frame bending.
- "Severe" = major collision, multiple areas, glass smashed, wheel pushed in, or clear structural risk.

STEP 7: OUTPUT FORMAT (IMPORTANT)
Return STRICTLY this JSON object (no extra text, no markdown):

{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [ ... ],
  "damage_types": [ ... ],
  "recommended_repairs": [ ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean,
  "customer_summary": "1-3 sentence explanation in friendly language, in plain English, no emojis."
}
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build OpenAI content payload
    content: List[dict] = []
    main_text = (
        f"Analyze these vehicle damage photos for the shop '{shop.name}'. "
        "Follow the system instructions exactly and return ONLY the JSON."
    )
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    # Use at most 3 images per request to keep latency and cost sane
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
        raw = json.loads(raw_content)

        result = postprocess_estimate(raw, vin)
        return result

    except Exception as e:
        print("AI estimator error:", e)
        # Conservative fallback
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
                "damage that will need professional repair. A technician will "
                "confirm the exact cost in person."
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
# APPOINTMENT SLOTS (simple generator, per-shop)
# ============================================================

def get_appointment_slots(shop: ShopConfig, n: int = 3) -> List[datetime.datetime]:
    """
    Simple slot generator.
    In the future you can plug in a real calendar using shop.calendar_id.
    """
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
    Main Twilio SMS entrypoint (multi-shop).
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

        areas = ", ".join(result["damage_areas"]) or "specific panels detected"
        types = ", ".join(result["damage_types"]) or "detailed damage types detected"
        summary = result.get("customer_summary") or ""

        slots = get_appointment_slots(shop)
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
        "- Send 1–5 clear photos of the damage (full vehicle corners help).",
        "- Optional: include your 17-character VIN in the text.",
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
