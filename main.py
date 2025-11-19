import os
import json
import re
import uuid
import base64
import datetime
from typing import Dict, Optional, List

import httpx
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# -------------------------------------------------------------------
# Environment & basic setup
# -------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

app = FastAPI()

# Twilio client (used only for media download)
twilio_client: Optional[Client] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -------------------------------------------------------------------
# Models & DB
# -------------------------------------------------------------------


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
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


# -------------------------------------------------------------------
# Shop configuration (multi-shop via token)
# -------------------------------------------------------------------


class ShopConfig(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """Load shops from SHOPS_JSON or fall back to a single default shop."""
    if not SHOPS_JSON:
        default = ShopConfig(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )
        return {default.webhook_token: default}

    raw = json.loads(SHOPS_JSON)
    shops: Dict[str, ShopConfig] = {}
    for item in raw:
        shop = ShopConfig(**item)
        shops[shop.webhook_token] = shop
    return shops


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()


def get_shop(request: Request) -> ShopConfig:
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


# -------------------------------------------------------------------
# Simple in-memory session store for SMS flows
# -------------------------------------------------------------------

# Keyed by f"{shop.id}:{phone}"
# Example values:
# {
#   "stage": "await_area_confirm" | "await_booking",
#   "image_urls": [...],
#   "vin": "1ABCDEFG...",
#   "areas": [...],
#   "slots": [datetime, ...]
# }
SESSIONS: Dict[str, dict] = {}

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    """Grab all Twilio MediaUrlX values from the incoming form."""
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
    m = VIN_PATTERN.search(text.upper())
    if m:
        return m.group(1)
    return None


def download_twilio_image(media_url: str) -> bytes:
    """Download media from Twilio's CDN using the REST client."""
    if not twilio_client:
        raise RuntimeError("Twilio client not configured")

    # Use the underlying HTTP client on Twilio's rest client
    # (this works in Twilio's SDK; Railway runtime already has the env vars)
    resp = twilio_client.http_client.request("GET", media_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download media: {resp.status_code}")
    return resp.content


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def call_openai_json(system_prompt: str, content: List[dict], max_tokens: int = 900) -> dict:
    """Call OpenAI chat completion with JSON response."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return json.loads(raw)


# -------------------------------------------------------------------
# Panel / area pre-scan (Step 1)
# -------------------------------------------------------------------

async def analyze_damage_areas(image_urls: List[str], shop: ShopConfig) -> dict:
    """
    Step 1 – ONLY identify visibly damaged panels/areas.

    This is deliberately conservative: it should list ONLY the panels that
    clearly look damaged in the photos. No guessing.
    """
    if not OPENAI_API_KEY:
        return {
            "detected_areas": [],
            "uncertain": True,
            "notes": "AI key not configured; cannot auto-detect panels.",
        }

    system_prompt = """
You are a professional collision damage estimator. Your ONLY job in this step
is to look at the photos and identify which EXTERIOR PANELS OR AREAS
VISIBLY show damage.

STRICT RULES:

- Only list panels/areas that clearly show dents, deformation, scrapes, cracks,
  broken glass, bent metal/plastic, or obvious misalignment.
- DO NOT guess about hidden or possible damage.
- DO NOT list panels that are outside the frame or look normal.
- Be conservative: when in doubt, leave it out.

USE THIS STANDARD PANEL VOCABULARY ONLY (when applicable):

FRONT:
- front bumper upper
- front bumper lower
- grille
- hood
- left front fender
- right front fender
- left headlight
- right headlight
- front left fog light / DRL
- front right fog light / DRL

REAR:
- rear bumper upper
- rear bumper lower
- trunk lid / tailgate
- left quarter panel
- right quarter panel
- left taillight
- right taillight

SIDES / DOORS / ROCKERS:
- left front door
- left rear door
- right front door
- right rear door
- left rocker panel / sill
- right rocker panel / sill
- A-pillar left
- A-pillar right
- B-pillar left
- B-pillar right
- C-pillar left
- C-pillar right
- left side mirror
- right side mirror

ROOF & GLASS:
- roof panel
- sunroof / moonroof
- windshield glass
- windshield frame / header
- back glass
- left front window glass
- left rear window glass
- right front window glass
- right rear window glass

WHEELS & TIRES:
- left front wheel / rim
- right front wheel / rim
- left rear wheel / rim
- right rear wheel / rim
- left front tire
- right front tire
- left rear tire
- right rear tire

OUTPUT JSON EXACTLY:

{
  "detected_areas": [
    "left front door",
    "left rocker panel / sill"
  ],
  "uncertain": false,
  "notes": "Short note about any uncertainty or limitations."
}
"""

    content: List[dict] = []
    content.append(
        {
            "type": "text",
            "text": (
                f"These are customer damage photos for {shop.name}.\n"
                "Identify ONLY the panels/areas that visibly show damage. "
                "Do NOT estimate cost yet."
            ),
        }
    )

    usable_images = image_urls[:8]
    for url in usable_images:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img_bytes = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img_bytes)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception:
                # If media download fails, just skip that image
                continue
        else:
            content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        result = await call_openai_json(system_prompt, content)
    except Exception:
        result = {
            "detected_areas": [],
            "uncertain": True,
            "notes": "Error while calling AI for panel detection.",
        }

    # Ensure keys exist
    result.setdefault("detected_areas", [])
    result.setdefault("uncertain", True)
    result.setdefault("notes", "")

    # Normalise to list of strings
    cleaned_areas = []
    for a in result["detected_areas"]:
        if isinstance(a, str):
            cleaned_areas.append(a.strip())
    result["detected_areas"] = cleaned_areas

    return result


# -------------------------------------------------------------------
# Full estimate (Step 2)
# -------------------------------------------------------------------

SEVERITY_ORDER = {"Minor": 0, "Moderate": 1, "Severe": 2}


def normalize_severity(raw: Optional[str]) -> str:
    if not raw:
        return "Moderate"
    text = raw.strip().lower()
    if text.startswith("min"):
        return "Minor"
    if text.startswith("sev") or "total" in text:
        return "Severe"
    return "Moderate"


def is_wheel_only(areas: List[str]) -> bool:
    """True if all confirmed areas are wheels/tires and no body/glass panels."""
    if not areas:
        return False
    wheel_keywords = ["wheel / rim", "tire"]
    body_keywords = [
        "bumper",
        "fender",
        "door",
        "quarter panel",
        "hood",
        "trunk",
        "tailgate",
        "rocker",
        "pillar",
        "mirror",
        "glass",
        "window",
        "windshield",
        "back glass",
        "headlight",
        "taillight",
        "roof",
        "grille",
    ]
    has_wheel = False
    has_body = False
    for a in areas:
        lower = a.lower()
        if any(wk in lower for wk in wheel_keywords):
            has_wheel = True
        if any(bk in lower for bk in body_keywords):
            has_body = True
    return has_wheel and not has_body


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def postprocess_estimate(raw: dict, confirmed_areas: List[str]) -> dict:
    """
    Safety rails on top of the model output:
    - Force areas to the confirmed list from Step 1.
    - Normalise severity.
    - Clamp insane wheel-only estimates.
    - Ensure cost range is consistent.
    """
    result = dict(raw)

    # 1) Damage areas: trust the confirmed panels, not the model.
    if confirmed_areas:
        result["damage_areas"] = confirmed_areas
    else:
        result["damage_areas"] = [
            a.strip() for a in result.get("damage_areas", []) if isinstance(a, str)
        ]

    # 2) Severity normalisation
    result["severity"] = normalize_severity(result.get("severity"))

    # 3) Cost range clean-up
    min_cost = result.get("min_cost")
    max_cost = result.get("max_cost")

    # If model didn't give numbers, base on severity
    if not isinstance(min_cost, (int, float)) or not isinstance(
        max_cost, (int, float)
    ):
        if result["severity"] == "Minor":
            min_cost, max_cost = 150.0, 900.0
        elif result["severity"] == "Moderate":
            min_cost, max_cost = 900.0, 3500.0
        else:
            min_cost, max_cost = 3500.0, 9000.0

    if min_cost > max_cost:
        min_cost, max_cost = max_cost, min_cost

    # Wheel-only sanity clamp
    if is_wheel_only(result["damage_areas"]):
        # For pure rim / curb-rash jobs in Ontario, we're somewhere in this band.
        min_cost = clamp(min_cost, 150.0, 900.0)
        max_cost = clamp(max_cost, min_cost + 50.0, 1300.0)
        # And severity should not be "Severe" for cosmetic wheel damage
        if result["severity"] == "Severe":
            result["severity"] = "Moderate"

    result["min_cost"] = float(min_cost)
    result["max_cost"] = float(max_cost)

    # Confidence default
    conf = result.get("confidence")
    if not isinstance(conf, (int, float)):
        conf = 0.7
    result["confidence"] = float(conf)

    # Customer summary fallback
    if not result.get("customer_summary"):
        result["customer_summary"] = (
            "Preliminary AI estimate based on visible exterior damage. "
            "An in-person inspection may change the final cost."
        )

    # Recommended repairs / damage types normalisation
    result["damage_types"] = [
        t.strip() for t in result.get("damage_types", []) if isinstance(t, str)
    ]
    result["recommended_repairs"] = [
        r.strip() for r in result.get("recommended_repairs", []) if isinstance(r, str)
    ]

    return result


async def estimate_damage_from_images(
    image_urls: List[str],
    shop: ShopConfig,
    vin: Optional[str],
    confirmed_areas: List[str],
) -> dict:
    """
    Step 2 – full estimate, but with panels locked to confirmed_areas.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    system_prompt = """
You are a senior collision estimator in Ontario with 10+ years of experience.
You are helping a body shop do a **preliminary, non-binding AI estimate** from
photos and an optional VIN.

Very important:

- The caller will supply a list of CONFIRMED PANELS/AREAS that the customer
  has verified as damaged.
- You MUST keep your estimate **consistent** with those confirmed areas.
- You may refine severity and damage TYPES, but you MUST NOT invent extra
  panels or areas outside the confirmed list.
- If you think more damage probably exists, mention it in the summary, but do
  not list unconfirmed panels in the structured area list.

Ontario-style estimating assumptions:
- Labour, materials, and refinish costs are roughly 2025 Ontario market rates.
- Use a realistic cost band (min_cost–max_cost), not a single number.
- Remember that replacing modern headlights/taillights, sensors, and cameras
  is expensive; simple scrapes are not.

Output MUST be valid JSON with this shape:

{
  "severity": "Minor | Moderate | Severe",
  "damage_areas": ["..."],
  "damage_types": ["dent", "paint scratch", "crack", "plastic tear", ...],
  "recommended_repairs": [
    "repair and refinish left front door",
    "replace rear bumper cover",
    "calibrate front radar sensor"
  ],
  "impact_profile": "rear-end | front-end | side-swipe | parking impact | rollover | unknown",
  "safety_flags": [
    "Possible sensor or ADAS impact – recommend scan",
    "Possible structural damage – recommend full frame / unibody measurement"
  ],
  "min_cost": 3000.0,
  "max_cost": 5200.0,
  "confidence": 0.0_to_1.0,
  "customer_summary": "Short, customer-friendly explanation (2–4 sentences)."
}

Rules:
- Be honest about uncertainty; use lower confidence when photos are limited.
- Never promise that this is a final repair bill – always treat it as preliminary.
- If the photos don't clearly show something, DON'T guess exact parts; keep it generic.
"""

    # Build user content
    text_parts = [
        f"Shop: {shop.name}",
        "",
        "CONFIRMED DAMAGED AREAS (from previous step):",
    ]
    if confirmed_areas:
        for a in confirmed_areas:
            text_parts.append(f"- {a}")
    else:
        text_parts.append("- (none clearly confirmed; infer gently from visible damage only)")

    if vin:
        text_parts.append("")
        text_parts.append(f"Customer provided VIN (may help with options): {vin}")

    text_parts.append("")
    text_parts.append(
        "Now produce a detailed preliminary estimate in JSON as per the schema. "
        "Remember: DO NOT add new panels that are not in the confirmed list."
    )

    content: List[dict] = [{"type": "text", "text": "\n".join(text_parts)}]

    usable_images = image_urls[:8]
    for url in usable_images:
        if url.startswith("https://api.twilio.com") and twilio_client:
            try:
                img_bytes = download_twilio_image(url)
                data_url = image_bytes_to_data_url(img_bytes)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception:
                continue
        else:
            content.append({"type": "image_url", "image_url": {"url": url}})

    raw = await call_openai_json(system_prompt, content)
    return postprocess_estimate(raw, confirmed_areas)


# -------------------------------------------------------------------
# DB + scheduling helpers
# -------------------------------------------------------------------

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
            customer_summary=result.get("customer_summary"),
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


def get_appointment_slots(n: int = 3) -> List[datetime.datetime]:
    """Very simple: tomorrow at 09:00, 11:00, 14:00 local time."""
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14]
    slots: List[datetime.datetime] = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


def format_slot(dt: datetime.datetime) -> str:
    return dt.strftime("%a %b %d at %I:%M %p")


# -------------------------------------------------------------------
# Public endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    """
    Main Twilio webhook.

    Stages:
      - No session + images   -> Step 1: panel pre-scan, ask for confirmation (1/2)
      - stage=await_area_confirm + text reply -> run Step 2 or reset
      - stage=await_booking + text reply -> confirm appointment choice
    """
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    if not from_number:
        reply.message("Missing phone number in request.")
        return Response(content=str(reply), media_type="application/xml")

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # --------------------------------------------------------------
    # Stage: booking selection (customer replies 1/2/3)
    # --------------------------------------------------------------
    if session and session.get("stage") == "await_booking" and not image_urls:
        choice = body.strip()
        slots: List[datetime.datetime] = session.get("slots") or []

        if choice in {"1", "2", "3"}:
            idx = int(choice) - 1
            if 0 <= idx < len(slots):
                chosen = slots[idx]
                reply.message(
                    f"All set! We've reserved an in-person estimate for "
                    f"{format_slot(chosen)} at {shop.name}. "
                    "If you need to reschedule, please call the shop directly."
                )
                SESSIONS.pop(session_key, None)
                return Response(content=str(reply), media_type="application/xml")

        # If we reach here, the input was invalid
        lines = [
            "Please reply with one of the numbers for your preferred time:",
        ]
        for i, s in enumerate(slots, start=1):
            lines.append(f"{i}) {format_slot(s)}")
        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # --------------------------------------------------------------
    # Stage: panel confirmation (customer replies 1/2)
    # --------------------------------------------------------------
    if session and session.get("stage") == "await_area_confirm" and not image_urls:
        choice = body.lower()
        confirmed_areas: List[str] = session.get("areas", [])
        stored_images: List[str] = session.get("image_urls", [])
        stored_vin: Optional[str] = session.get("vin")

        if choice in {"1", "yes", "y"}:
            # Run full estimate with confirmed panels
            try:
                result = await estimate_damage_from_images(
                    stored_images,
                    shop=shop,
                    vin=stored_vin,
                    confirmed_areas=confirmed_areas,
                )
            except Exception:
                reply.message(
                    "Sorry, there was an error generating your estimate. "
                    "Please try again in a few minutes."
                )
                SESSIONS.pop(session_key, None)
                return Response(content=str(reply), media_type="application/xml")

            estimate_id = save_estimate_to_db(
                shop=shop,
                phone=from_number,
                vin=stored_vin,
                result=result,
            )

            slots = get_appointment_slots()
            SESSIONS[session_key] = {
                "stage": "await_booking",
                "slots": slots,
            }

            # Build customer-facing message
            min_cost = result.get("min_cost")
            max_cost = result.get("max_cost")
            cost_line = "Estimated Cost (Ontario 2025):"
            if isinstance(min_cost, (int, float)) and isinstance(
                max_cost, (int, float)
            ):
                cost_line += f" ${int(min_cost):,} – ${int(max_cost):,}"
            else:
                cost_line += " not clearly determined"

            lines: List[str] = [
                f"AI Damage Estimate for {shop.name}",
                "",
                f"Severity: {result.get('severity', 'Unknown')}",
                cost_line,
            ]

            areas = result.get("damage_areas") or []
            if areas:
                lines.append(
                    "Areas: " + ", ".join(areas)
                )

            dmg_types = result.get("damage_types") or []
            if dmg_types:
                lines.append(
                    "Damage Types: " + ", ".join(dmg_types)
                )

            summary = result.get("customer_summary")
            if summary:
                lines.append("")
                lines.append(summary)

            lines.append("")
            lines.append(f"Estimate ID (internal): {estimate_id}")

            # Appointment options
            if slots:
                lines.append("")
                lines.append("Reply with a number to book an in-person estimate:")
                for i, s in enumerate(slots, start=1):
                    lines.append(f"{i}) {format_slot(s)}")

            reply.message("\n".join(lines))
            return Response(content=str(reply), media_type="application/xml")

        elif choice in {"2", "no", "n"}:
            reply.message(
                "No problem. Please send 1–5 clear photos focusing on the damaged "
                "area(s) of your vehicle, and I'll scan them again."
            )
            SESSIONS.pop(session_key, None)
            return Response(content=str(reply), media_type="application/xml")

        else:
            reply.message(
                "Please reply 1 if the detected areas look roughly correct, or 2 if "
                "they are wrong and you'll resend clearer photos."
            )
            return Response(content=str(reply), media_type="application/xml")

    # --------------------------------------------------------------
    # New photos received – start Step 1 (panel pre-scan)
    # --------------------------------------------------------------
    if image_urls:
        # Any new image resets the previous session
        SESSIONS.pop(session_key, None)

        panel_result = await analyze_damage_areas(image_urls, shop)
        areas = panel_result.get("detected_areas", [])
        uncertain = panel_result.get("uncertain", True)
        notes = panel_result.get("notes") or ""

        lines: List[str] = [
            f"AI Pre-Scan for {shop.name}",
            "",
        ]
        if areas:
            lines.append("I can see damage in these areas:")
            for a in areas:
                lines.append(f"- {a}")
        else:
            lines.append(
                "I couldn't clearly detect specific panels from these photos yet."
            )

        if notes:
            lines.append("")
            lines.append(f"Notes: {notes}")

        if uncertain:
            lines.append("")
            lines.append(
                "I'm being conservative here – I only list panels that clearly look damaged."
            )

        lines.append("")
        lines.append("Reply 1 if this list looks roughly correct.")
        lines.append("Reply 2 if it's wrong and you'll resend clearer photos.")

        SESSIONS[session_key] = {
            "stage": "await_area_confirm",
            "image_urls": image_urls,
            "vin": vin,
            "areas": areas,
        }

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # --------------------------------------------------------------
    # No images & no active session – send instructions
    # --------------------------------------------------------------
    lines = [
        f"Hi from {shop.name}!",
        "",
        "To get a fast AI damage estimate, please:",
        "- Send 1–5 clear photos of the damaged area (front, rear, side, wheels, roof, etc.)",
        "- Optional: include your 17-character VIN in the text",
    ]
    reply.message("\n".join(lines))
    return Response(content=str(reply), media_type="application/xml")
