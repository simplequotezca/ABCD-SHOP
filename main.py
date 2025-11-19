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
# SHOP CONFIG (multi-shop via SHOPS_JSON)
# ============================================================

class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: Optional[str] = None  # reserved for future use


def load_shops() -> Dict[str, ShopConfig]:
    """
    Parse SHOPS_JSON env var into token->ShopConfig map.

    Example SHOPS_JSON:

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
# POST-PROCESSING + SANITY RULES (ULTRA-STRICT OPTION C)
# ============================================================

def postprocess_estimate(raw: dict, vin: Optional[str]) -> dict:
    """
    Tighten and normalize the model output so it behaves like a
    professional estimator:

    - Enforce valid severity values
    - Ultra-strict collision overrides (Option C)
    - Wheel / curb-rash logic vs full collision
    - Ontario 2025 pricing bands by severity
    - Clamp crazy cost ranges
    """
    result = dict(raw) if raw else {}

    # --- Defaults / structure ------------------------------------------------
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
        "Based on the photos, this is a preliminary estimate. "
        "An in-person inspection may adjust the final cost.",
    )

    areas = [a for a in result.get("damage_areas") or [] if isinstance(a, str)]
    types_ = [t for t in result.get("damage_types") or [] if isinstance(t, str)]
    recs = [r for r in result.get("recommended_repairs") or [] if isinstance(r, str)]

    areas_l = [a.lower() for a in areas]
    types_l = [t.lower() for t in types_]
    recs_l = [r.lower() for r in recs]
    summary = result.get("customer_summary") or ""
    summary_l = summary.lower()

    num_panels = len(areas_l)

    # --- Feature flags -------------------------------------------------------

    wheel_related = any(
        kw in a
        for a in areas_l
        for kw in ["wheel", "rim", "alloy"]
    )

    bumper_present = any("bumper" in a for a in areas_l)
    fender_present = any("fender" in a for a in areas_l)
    hood_present = any("hood" in a for a in areas_l)
    quarter_present = any("quarter" in a for a in areas_l)
    door_present = any("door" in a for a in areas_l)
    headlight_present = any("headlight" in a for a in areas_l)
    taillight_present = any("taillight" in a for a in areas_l)

    structural_type = any(
        kw in types_l
        for kw in ["metal distortion", "misalignment", "frame", "structural"]
    )

    tear_crack_type = any(
        kw in types_l
        for kw in ["crack", "plastic tear", "bumper deformation"]
    )

    severe_repair = any(
        kw in recs_l
        for kw in [
            "panel replacement",
            "bumper replacement",
            "quarter panel replacement",
            "frame",
            "structural pull",
        ]
    )

    # Anything in here is a "major collision indicator"
    major_indicators = [
        structural_type,
        tear_crack_type,
        severe_repair,
        headlight_present,
        taillight_present,
        (bumper_present and fender_present),
        (bumper_present and hood_present),
        num_panels >= 3,
    ]
    has_major_indicator = any(major_indicators)

    # --- Severity normalization ---------------------------------------------

    severity = (result.get("severity") or "Moderate").title()
    if severity not in {"Minor", "Moderate", "Severe"}:
        severity = "Moderate"

    # If the model says Minor but there's obvious multi-panel or bumper damage → bump to Moderate
    if severity == "Minor" and (num_panels >= 2 or bumper_present or fender_present):
        severity = "Moderate"

    # ULTRA-STRICT OPTION C:
    # If ANY major indicator is present, force severity = Severe (top bucket).
    if has_major_indicator:
        severity = "Severe"

    # If the model somehow said Severe but everything is just tiny wheel rash,
    # soften to Minor with wheel band.
    if severity == "Severe" and wheel_related and not bumper_present and num_panels <= 2:
        severity = "Minor"

    # If there are NO areas and NO types, but the model still gave a number,
    # treat as "uncertain collision" → Moderate.
    if not areas_l and not types_l:
        severity = "Moderate"

    result["severity"] = severity

    # --- Severity-based base bands (Ontario 2025 retail CAD) -----------------
    # These are wide buckets that we will clamp the model into.

    if severity == "Minor":
        base_min, base_max = 250.0, 1500.0
    elif severity == "Moderate":
        base_min, base_max = 1200.0, 5500.0
    else:  # Severe
        base_min, base_max = 4500.0, 15000.0

    # Wheel-only or light wheel + small scuff: narrower band
    if wheel_related and not bumper_present and num_panels <= 2 and not has_major_indicator:
        # wheel refinishing / minor curb rash
        base_min, base_max = 250.0, 900.0

    # Bumper-only, light to medium
    if bumper_present and not has_major_indicator and num_panels <= 2 and not structural_type:
        # bumper repair + paint
        base_min = max(base_min, 600.0)
        base_max = min(base_max, 2200.0)

    # Multi-panel collision: increase upper bounds
    if num_panels > 2:
        base_min += (num_panels - 2) * 300.0
        base_max += (num_panels - 2) * 900.0

    # Very heavy combination of bumper + hood + fender / quarters → high side
    if severity == "Severe" and (
        (bumper_present and fender_present and hood_present)
        or quarter_present
    ):
        base_min = max(base_min, 6500.0)
        base_max = max(base_max, 9000.0)

    # --- Clamp model-proposed costs into these bands -------------------------

    try:
        min_c = float(result.get("min_cost", base_min))
        max_c = float(result.get("max_cost", base_max))
    except Exception:
        min_c, max_c = base_min, base_max

    # Swap if reversed
    if max_c < min_c:
        min_c, max_c = max_c, min_c

    # Pull extreme ranges back toward the center
    if max_c - min_c > 8000:
        mid = (min_c + max_c) / 2.0
        min_c = mid - 2500
        max_c = mid + 2500

    # Clamp to severity band
    min_c = max(min_c, base_min)
    max_c = min(max_c, base_max)

    # Ensure some spread
    if max_c - min_c < 150:
        max_c = min_c + 150

    result["min_cost"] = round(min_c)
    result["max_cost"] = round(max_c)

    # --- Final summary adjustments ------------------------------------------

    if not summary.strip():
        result["customer_summary"] = (
            "This is a preliminary AI estimate based on the photos. "
            "Labour, parts availability, and hidden damage may change the final cost. "
            "A licensed estimator will confirm everything in person."
        )
    elif "no visible damage" in summary_l and (bumper_present or fender_present or headlight_present):
        # Guardrail: if it claims "no visible damage" but panels are listed, rewrite tone.
        result["customer_summary"] = (
            "The photos show visible body damage to the listed panels. "
            "This range reflects typical repair and repaint costs in Ontario. "
            "A full teardown or frame measurement may reveal additional work."
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

    system_prompt = """
You are a **licensed Ontario (Canada) auto-body damage estimator in 2025**
with 15+ years of collision centre experience. You write estimates for
retail customers (not discounted DRP insurance rates).

You are given 1–5 photos of vehicle damage, and sometimes a VIN.

Your job is to behave like a professional estimator:
- Identify which exterior panels and components are damaged.
- Classify the type and severity of visible damage.
- Map that damage to realistic repair operations.
- Produce an Ontario-typical 2025 retail **labour + materials** price range.
- When in doubt, **never under-estimate** heavy collision damage.

You are NOT doing mechanical diagnostics. Focus on body, paint,
bumper, lamps, and obvious suspension / wheel impacts.

---------------------------------------------------------------
PANEL LIST (use only these terms)
---------------------------------------------------------------
Choose from, as needed:

- "front bumper upper"
- "front bumper lower"
- "rear bumper upper"
- "rear bumper lower"
- "left fender"
- "right fender"
- "left front door"
- "right front door"
- "left rear door"
- "right rear door"
- "hood"
- "trunk"
- "roof"
- "left quarter panel"
- "right quarter panel"
- "rocker panel"
- "grille area"
- "headlight area"
- "taillight area"
- "left front wheel / rim"
- "right front wheel / rim"
- "left rear wheel / rim"
- "right rear wheel / rim"

If something is clearly damaged, name the closest matching panel(s).
Never say "general damage" or "unspecified area".

---------------------------------------------------------------
DAMAGE TYPES (choose all that apply)
---------------------------------------------------------------
Use these exact strings:

- "dent"
- "crease dent"
- "sharp dent"
- "paint scratch"
- "deep scratch"
- "paint scuff"
- "paint transfer"
- "crack"
- "plastic tear"
- "bumper deformation"
- "metal distortion"
- "misalignment"
- "rust exposure"
- "curb rash"

"metal distortion" / "misalignment" should be reserved for stronger
impacts where metal or alignment is visibly off.

"curb rash" is only for wheel / rim damage from scraping a curb.

---------------------------------------------------------------
RECOMMENDED REPAIRS (operations)
---------------------------------------------------------------
Use a mix of these operations where appropriate:

- "PDR (paintless dent repair)"
- "panel repair + paint"
- "panel replacement"
- "bumper repair + paint"
- "bumper replacement"
- "refinish only (no structural repair)"
- "blend adjacent panels"
- "wheel refinish / curb rash repair"
- "wheel repair + refinish"
- "wheel replacement"
- "headlamp replacement"
- "taillamp replacement"
- "recalibration (sensors/cameras)"
- "frame / structural measurement"

If damage is light and clearly PDR-suitable, you may use PDR.
If plastic is cracked or torn, prefer replacement or proper plastic repair.

---------------------------------------------------------------
ONTARIO 2025 RETAIL PRICING GUIDELINE (CAD)
---------------------------------------------------------------
These are typical **retail** price bands (parts + labour) per operation:

- Wheel refinish / curb rash repair: 200 – 450
- Wheel replacement (single OEM): 450 – 1500
- PDR small dent: 150 – 600
- Panel repaint only: 350 – 900
- Panel repair + repaint: 600 – 1600
- Bumper repaint only: 400 – 900
- Bumper repair + repaint: 750 – 1400
- Bumper replacement (painted & installed): 800 – 2500
- Door replacement & refinish: 800 – 2200
- Fender repair + paint: 600 – 1900
- Quarter panel repair: 900 – 2500
- Quarter panel replacement: 1800 – 4800
- Hood repaint: 400 – 900
- Hood replacement & refinish: 600 – 2400
- Headlamp / taillamp replacement (each): 300 – 1400
- Structural / frame measurement & pull: 900 – 3500

Use these as building blocks:
- Minor cosmetic, one panel → low end of a single operation.
- Multiple panels or heavier hit → sum realistic operations.
- Luxury / EV / aluminum panels → bias 15–30% higher.
- NEVER give a tiny number for clearly heavy collision photos.

---------------------------------------------------------------
SEVERITY SCALE
---------------------------------------------------------------
Return severity ONLY as:
- "Minor"   → light cosmetic, small area, no structural
- "Moderate"→ multiple panels or larger repair, but mostly bolt-on / cosmetic
- "Severe"  → heavy collision, structural risk, lamps destroyed, or many panels

Rules:
- If headlamp or bumper is clearly smashed OR metal is badly distorted
  OR 3+ major panels are involved → severity must be "Severe".
- If you are unsure but it *looks bad*, choose "Severe", not "Minor".

---------------------------------------------------------------
VIN USAGE
---------------------------------------------------------------
If a VIN is provided:
- Use it ONLY to infer broad category (economy / mid-range / luxury / truck / EV).
- Do not decode specifics.
- Adjust costs: luxury / EV / aluminum → add roughly 15–30%.

Set "vin_used": true only if you actually adjusted based on the VIN.

---------------------------------------------------------------
UNCERTAIN OR LOW-CONFIDENCE CASES
---------------------------------------------------------------
If the photos are blurry, too far away, or mostly background:
- Do NOT say "no visible damage" when damage is clearly present.
- Instead, treat as at least "Moderate" and give a conservative but
  realistic band.
- Mention in the customer_summary that the photos are unclear and a
  proper in-person inspection is needed.

If the photos truly show no obvious damage:
- You may use a very low range like 50 – 150 and explain that nothing
  obvious is visible.

---------------------------------------------------------------
OUTPUT FORMAT (STRICT)
---------------------------------------------------------------
Think through the damage internally. Then output ONLY THIS JSON:

{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [ "front bumper lower", "right fender", ... ],
  "damage_types": [ "dent", "paint scuff", ... ],
  "recommended_repairs": [ "bumper repair + paint", "panel repair + paint", ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,  // 0.0–1.0
  "vin_used": boolean,
  "customer_summary": "1–3 sentence explanation in friendly, clear language"
}

Return valid JSON only. No extra commentary, no Markdown.
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build OpenAI content payload
    content: List[dict] = []
    main_text = (
        f"Analyze these vehicle damage photos for {shop.name} "
        "and follow the instructions carefully."
    )
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    # Use at most 3 images per request to keep latency/cost manageable
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
        "model": "gpt-4.1-mini",  # you can change to "gpt-4.1" for even higher quality
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
        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)

        result = postprocess_estimate(parsed, vin)
        return result

    except Exception as e:
        print("AI estimator error:", e)
        # Safe fallback
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 2500,
            "confidence": 0.4,
            "vin_used": bool(vin),
            "customer_summary": (
                "We had trouble analyzing the photos, but this looks like "
                "moderate damage. A licensed estimator will confirm the exact "
                "repairs and final cost in person."
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

        areas = ", ".join(result.get("damage_areas", [])) or "not clearly identified"
        types = ", ".join(result.get("damage_types", [])) or "not clearly identified"
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
