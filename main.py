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
    calendar_id: Optional[str] = None  # reserved for future Google Calendar link


def load_shops() -> Dict[str, ShopConfig]:
    """
    Parse SHOPS_JSON env var into token->ShopConfig map.

    SHOPS_JSON example:

    [
      { "id": "shop1", "name": "Brampton Auto Body", "webhook_token": "brampton123" },
      { "id": "shop2", "name": "Mississauga Collision Centre", "webhook_token": "miss_centre_456" }
    ]
    """
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
# PANEL GROUPING + COST CALIBRATION HELPERS
# ============================================================

# Map panel names (substrings) to high-level zones
PANEL_ZONES = {
    "front bumper": "front",
    "rear bumper": "rear",
    "hood": "front",
    "grille": "front",
    "headlight": "front",
    "taillight": "rear",
    "trunk": "rear",
    "decklid": "rear",
    "roof": "roof",
    "windshield": "glass_front",
    "back glass": "glass_rear",
    "rear glass": "glass_rear",
    "quarter panel": "rear_side",
    "fender": "front_side",
    "rocker": "side_lower",
    "sill": "side_lower",
    "door": "side",
    "pillar": "structure",
    "wheel": "wheel",
    "rim": "wheel",
    "tire": "wheel",
    "sunroof": "roof",
    "moonroof": "roof",
}

WHEEL_KEYWORDS = ["wheel", "rim", "tire"]
ROOF_KEYWORDS = ["roof", "roof rail", "sunroof", "moonroof"]


def classify_panel_zone(name: str) -> str:
    n = name.lower()
    for key, zone in PANEL_ZONES.items():
        if key in n:
            return zone
    return "unknown"


def contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def list_contains_any(items: List[str], keywords: List[str]) -> bool:
    return any(contains_any(x, keywords) for x in items)


def postprocess_cost_and_panels(
    raw: dict,
    vin: Optional[str],
    image_count: int,
) -> dict:
    """
    Final "brains" layer:
    - cleans up weird panel mixes (front vs rear vs roof vs wheel)
    - tightens cost ranges for simple jobs
    - clamps obviously insane values
    """

    # --- Basic defaults ---
    severity = (raw.get("severity") or "Moderate").title()
    damage_areas: List[str] = [a.strip() for a in raw.get("damage_areas", []) if a]
    damage_types: List[str] = [t.strip() for t in raw.get("damage_types", []) if t]
    recommended_repairs: List[str] = [
        r.strip() for r in raw.get("recommended_repairs", []) if r
    ]

    # --- Zone voting (front / rear / side / roof / wheel etc.) ---
    zone_counts: Dict[str, int] = {}
    for a in damage_areas:
        z = classify_panel_zone(a)
        zone_counts[z] = zone_counts.get(z, 0) + 1

    dominant_zone = None
    if zone_counts:
        dominant_zone = max(zone_counts.items(), key=lambda x: x[1])[0]

    # --- Heuristic: if we clearly have wheels only, drop random bumpers, etc. ---
    if damage_areas:
        wheel_only_image = list_contains_any(damage_areas, WHEEL_KEYWORDS) and not (
            list_contains_any(damage_areas, ROOF_KEYWORDS)
        )

        if wheel_only_image:
            # If it's basically curb rash, we expect only wheel / tire related panels.
            filtered = []
            for a in damage_areas:
                if list_contains_any([a], WHEEL_KEYWORDS):
                    filtered.append(a)
            if filtered:
                damage_areas = filtered

    # --- Heuristic: if roof dominates, suppress unrelated rear/front panels ---
    if dominant_zone == "roof":
        filtered = []
        for a in damage_areas:
            z = classify_panel_zone(a)
            if z in ("roof", "glass_front", "glass_rear", "structure"):
                filtered.append(a)
        if filtered:
            damage_areas = filtered

    # --- Rear vs front dominance cleanup ---
    if zone_counts.get("rear", 0) >= 2 and zone_counts.get("front", 0) == 1:
        # Probably a rear impact with a spurious "front bumper" mention.
        damage_areas = [
            a for a in damage_areas if classify_panel_zone(a) != "front"
        ]
    if zone_counts.get("front", 0) >= 2 and zone_counts.get("rear", 0) == 1:
        damage_areas = [
            a for a in damage_areas if classify_panel_zone(a) != "rear"
        ]

    # --- Cost sanity & specialization ---
    try:
        min_c = float(raw.get("min_cost", 600))
        max_c = float(raw.get("max_cost", 1500))
    except Exception:
        min_c, max_c = 600.0, 1500.0

    # Ensure min <= max
    if max_c < min_c:
        min_c, max_c = max_c, min_c

    # Global clamp to keep values sane
    # (Sub-100 is unrealistic; 20k is overkill for SMS pre-estimates)
    min_c = max(80.0, min_c)
    max_c = min(20000.0, max_c)

    # --- Special case: curb rash / wheel focus ---
    if list_contains_any(damage_areas, WHEEL_KEYWORDS) and len(damage_areas) <= 2:
        # Estimate for wheel refinishing / minor tire replacement.
        if severity.lower() == "minor":
            min_c, max_c = 150.0, 450.0
        elif severity.lower() == "moderate":
            min_c, max_c = 300.0, 900.0
        else:
            min_c, max_c = 400.0, 1200.0

    # --- Special case: roof-only or roof + glass ---
    if list_contains_any(damage_areas, ROOF_KEYWORDS) and not list_contains_any(
        damage_areas, WHEEL_KEYWORDS
    ):
        if severity.lower() == "minor":
            min_c, max_c = 600.0, 1600.0
        elif severity.lower() == "moderate":
            min_c, max_c = 1200.0, 2800.0
        else:
            min_c, max_c = 2000.0, 4800.0

    # --- Special case: heavy rear or front smash (>=3 panels, severe) ---
    rear_panels = [a for a in damage_areas if classify_panel_zone(a) in ("rear", "rear_side", "glass_rear")]
    front_panels = [a for a in damage_areas if classify_panel_zone(a) in ("front", "front_side", "glass_front")]

    if severity.lower() == "severe" and (len(rear_panels) >= 3 or len(front_panels) >= 3):
        # Big collision; ensure we are not underpricing.
        min_c = max(min_c, 2500.0)
        max_c = max(max_c, min_c + 800.0)

    # --- Reasonable spread: keep range within 5k unless truly severe ---
    spread = max_c - min_c
    if spread > 5000.0:
        mid = (min_c + max_c) / 2
        min_c = max(100.0, mid - 2500.0)
        max_c = mid + 2500.0

    # Round to nearest 10 for nicer SMS display
    min_c = round(min_c / 10.0) * 10.0
    max_c = round(max_c / 10.0) * 10.0

    result = dict(raw)
    result["severity"] = severity
    result["damage_areas"] = damage_areas
    result["damage_types"] = damage_types
    result["recommended_repairs"] = recommended_repairs
    result["min_cost"] = float(min_c)
    result["max_cost"] = float(max_c)
    return result


# ============================================================
# AI DAMAGE ESTIMATION (ONTARIO 2025, MULTI-IMAGE, ULTRA PROMPT)
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

    # Ultra-detailed system prompt: multi-image, full vehicle coverage, strict no-hallucination
    system_prompt = """
You are a MASTER collision estimator in Ontario, Canada, year 2025,
with 15+ years of professional auto body and collision repair experience.

Your job: from PHOTOS ONLY (and an optional VIN), produce a realistic, shop-ready
pre-estimate for a retail customer (NO deep insurance discounts).

You must be:
- CONSERVATIVE but REALISTIC
- Extremely careful to NOT invent panels or damage you cannot clearly see
- Very strict about front vs rear vs roof vs side
- Able to handle: sedans, coupes, hatchbacks, SUVs, vans, and pickup trucks

You will receive 1–5 photos of a vehicle.
Sometimes they are close-ups (e.g., wheel, door corner, roof dent).
Sometimes they are wider shots (front end crash, rear end crash, etc).

-------------------------------------------------------------------------------
STEP 0: GLOBAL PRINCIPLES
-------------------------------------------------------------------------------
1) NEVER GUESS panels that are not clearly visible.
   - If you see only a rear bumper and trunk → DO NOT mention front bumper.
   - If the photo is only a wheel close-up → DO NOT mention fenders or bumpers.
   - If the photo is only roof and a bit of glass → DO NOT mention doors or bumpers.
2) If unsure whether a panel is actually damaged, LEAVE IT OUT.
   - It is BETTER to UNDER-LIST panels than to hallucinate new ones.
3) Each damaged panel must be a REAL, specific automotive area.

-------------------------------------------------------------------------------
STEP 1: PER-IMAGE ANALYSIS (YOU DO THIS INTERNALLY)
-------------------------------------------------------------------------------
For each image you receive, you internally identify:
- which panels are visible
- which panels clearly show damage
- how large the damage is on that panel (approx % of surface)
- what types of damage occur (dent, scratch, crack, scuff, rust, etc.)
You then MERGE these into a single global picture of the vehicle damage,
avoiding double-counting the same dent across different angles.

-------------------------------------------------------------------------------
STEP 2: ALLOWED PANEL NAMES (USE THESE WORDINGS)
-------------------------------------------------------------------------------
When you output damage_areas, you must select from / adapt to the closest of:

FRONT / REAR BUMPERS:
- front bumper upper
- front bumper lower
- rear bumper upper
- rear bumper lower

HOOD / TRUNK / ROOF:
- hood
- trunk lid / decklid
- roof
- roof rail left
- roof rail right

FRONT & REAR STRUCTURE / LIGHTS:
- grille area
- left headlight area
- right headlight area
- left fog light area
- right fog light area
- left taillight area
- right taillight area

SIDE PANELS:
- left fender
- right fender
- left front door outer panel
- right front door outer panel
- left rear door outer panel
- right rear door outer panel
- left quarter panel
- right quarter panel
- left rocker panel / sill
- right rocker panel / sill
- left A-pillar
- right A-pillar
- left B-pillar
- right B-pillar
- left C-pillar
- right C-pillar

WHEELS / TIRES:
- left front wheel / rim
- right front wheel / rim
- left rear wheel / rim
- right rear wheel / rim
- left front tire
- right front tire
- left rear tire
- right rear tire

GLASS:
- windshield
- back glass / rear window
- left front door glass
- right front door glass
- left rear door glass
- right rear door glass
- left quarter glass
- right quarter glass
- sunroof / moonroof glass

MIRRORS:
- left side mirror
- right side mirror

If a very close panel does not fit exactly, choose the CLOSEST accurate label.
Do NOT invent “generic” labels like “general body damage”.

-------------------------------------------------------------------------------
STEP 3: ALLOWED DAMAGE TYPES
-------------------------------------------------------------------------------
Use one or more of:

- dent
- deep dent
- crease dent
- sharp dent
- paint scratch
- deep scratch
- paint scuff
- paint transfer
- crack
- glass crack
- glass chip
- plastic tear
- bumper deformation
- metal distortion
- misalignment
- rust exposure
- panel distortion
- hole / puncture
- peeling clearcoat

If you cannot clearly see a given type, DO NOT list it.

-------------------------------------------------------------------------------
STEP 4: DETERMINE IMPACT PATTERN (INTERNAL REASONING)
-------------------------------------------------------------------------------
From all images together, infer the MAIN impact pattern:

impact_location (choose best fit):
- "front"
- "rear"
- "left_side"
- "right_side"
- "roof"
- "multi_zone"

impact_type (free text, but concise), e.g.:
- "rear-end collision"
- "front-end collision"
- "side swipe"
- "parking pole impact"
- "curb impact on wheel"
- "object falling on roof"
- "hail-type roof damage"
- "minor parking scrape"

Use this only to guide severity and repair strategy.
Do NOT hallucinate other panels just because of impact_type.

-------------------------------------------------------------------------------
STEP 5: REPAIR OPERATIONS (SHOP LANGUAGE)
-------------------------------------------------------------------------------
Use any that apply:

- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- blend adjacent panels
- refinish wheel
- wheel replacement
- glass chip repair
- glass replacement
- headlight replacement
- taillight replacement
- mirror replacement
- roof repair + repaint
- trunk lid repair + repaint
- hood repair + repaint
- sensor / camera recalibration
- structural inspection

-------------------------------------------------------------------------------
STEP 6: ONTARIO 2025 RETAIL PRICING (CAD, CUSTOMER PAY)
-------------------------------------------------------------------------------
Use typical Ontario BODY SHOP retail pricing ranges (not insurance discounts):

ROUGH UNIT PRICES:
- PDR, small: 150–400
- PDR, larger/complex: 300–800
- Panel repaint (no repair): 350–900
- Panel repair + repaint: 600–1,600
- Bumper repaint only: 400–900
- Bumper repair + repaint: 750–1,400
- Bumper replacement + paint + install: 1,200–2,800
- Door repair + repaint: 700–1,800
- Door replacement + paint + install: 1,500–3,200
- Quarter panel repair: 900–2,500
- Quarter panel replacement: 1,800–4,800
- Hood repaint: 400–1,000
- Hood replacement: 700–2,400
- Roof repair + repaint: 1,200–4,000
- Roof replacement (rare, heavy): 3,000–8,000
- Wheel refinish (per wheel): 180–450
- Wheel replacement (per wheel, typical alloy): 450–900
- Windshield replacement: 450–1,100
- Back glass replacement: 500–1,200
- Door glass replacement: 350–800
- Headlight replacement (each, with aiming): 300–900
- Taillight replacement (each): 220–650
- Sensor/camera recalibration package: 250–600

RULES:
- Use realistic combinations of these operations.
- Minor cosmetic = low end of ranges.
- Clearly severe or multiple panels = high end or sum of multiple operations.
- If VIN or brand suggests luxury/EV/aluminum, bias 15–30% higher.
- DO NOT exceed 20,000 CAD total for an SMS pre-estimate.
- DO NOT go below 80 CAD.

-------------------------------------------------------------------------------
STEP 7: SEVERITY
-------------------------------------------------------------------------------
Choose ONE:

- "Minor"   = cosmetic, limited area, no obvious structural or glass breakage
- "Moderate"= multiple panels OR deeper dents OR some plastic tears / cracks
- "Severe"  = heavy deformation, multiple major panels, significant glass or structure

-------------------------------------------------------------------------------
STEP 8: VIN HANDLING
-------------------------------------------------------------------------------
If a VIN is provided, infer only broad category:
- economy
- mid-range
- luxury
- pickup / truck
- EV

Adjust the estimate level slightly (economy → lower range, luxury/EV → higher).

-------------------------------------------------------------------------------
STEP 9: ENSEMBLE ESTIMATE (INTERNAL)
-------------------------------------------------------------------------------
Internally, you imagine three estimators:
- Estimator A: conservative lower-bound
- Estimator B: typical mid-range
- Estimator C: high-end shop pricing

You then output:
- min_cost = near A
- max_cost = near C
But keep them realistic and consistent with operations and severity.

-------------------------------------------------------------------------------
STEP 10: OUTPUT FORMAT (MUST BE STRICT JSON)
-------------------------------------------------------------------------------
Return ONLY a single JSON object with exactly these keys:

{
  "severity": "Minor | Moderate | Severe",
  "damage_areas": [ "...", "..." ],
  "damage_types": [ "...", "..." ],
  "recommended_repairs": [ "...", "..." ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,      // 0.0–1.0, how confident you are overall
  "vin_used": boolean,       // true if VIN influenced pricing
  "impact_location": "front|rear|left_side|right_side|roof|multi_zone",
  "impact_type": "short phrase describing the scenario",
  "customer_summary": "1–3 sentence friendly explanation for the vehicle owner"
}

IMPORTANT:
- JSON ONLY. No extra commentary, no markdown.
- Only include panels and damage you CLEARLY SEE in the provided images.
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build OpenAI content payload
    content: List[dict] = []
    main_text = (
        f"Analyze these vehicle damage photos for {shop.name}. "
        "Follow the multi-step instructions carefully and return ONLY the JSON object."
    )
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    # Use at most 3 images per request to balance cost vs accuracy
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

        # Defaults + sanity
        raw.setdefault("severity", "Moderate")
        raw.setdefault("damage_areas", [])
        raw.setdefault("damage_types", [])
        raw.setdefault("recommended_repairs", [])
        raw.setdefault("min_cost", 600)
        raw.setdefault("max_cost", 1500)
        raw.setdefault("confidence", 0.7)
        raw.setdefault("vin_used", bool(vin))
        raw.setdefault("impact_location", "multi_zone")
        raw.setdefault("impact_type", "unspecified impact")
        raw.setdefault(
            "customer_summary",
            "Based on the photos, we estimate moderate damage. "
            "A detailed in-person inspection may adjust the final cost.",
        )

        # Final Python-level post-processing + cost calibration
        result = postprocess_cost_and_panels(
            raw=raw,
            vin=vin,
            image_count=len(usable_urls),
        )

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
            "impact_location": "multi_zone",
            "impact_type": "unspecified impact",
            "customer_summary": (
                "We had trouble analyzing the photos, but this looks like "
                "moderate damage. A technician will confirm in person."
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
    hours = [9, 11, 14]

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

        areas = ", ".join(result["damage_areas"]) or "No specific damaged panels clearly detected"
        types = ", ".join(result["damage_types"]) or "No specific damage types clearly detected"
        summary = result.get("customer_summary") or ""
        impact_location = result.get("impact_location", "multi_zone")
        impact_type = result.get("impact_type", "").strip()

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

        if impact_type:
            lines.append(f"Impact: {impact_type} ({impact_location})")

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
        "- Send 1–5 clear photos of the damage (different angles if possible)",
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
