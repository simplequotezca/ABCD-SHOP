import os
import json
import re
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

# ============================================================
# FastAPI + OpenAI setup
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Multi-shop config (tokenized routing via SHOPS_JSON)
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str  # used as ?token=... in Twilio URL


def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        # Safe default for local testing
        default = Shop(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )
        return {default.webhook_token: default}

    data = json.loads(raw)
    by_token: Dict[str, Shop] = {}
    for item in data:
        shop = Shop(**item)
        by_token[shop.webhook_token] = shop
    return by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()

# ============================================================
# In-memory session store
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_MINUTES = 60


def session_key(shop: Shop, phone: str) -> str:
    return f"{shop.id}:{phone}"


def cleanup_sessions() -> None:
    now = datetime.utcnow()
    expired = []
    for k, v in SESSIONS.items():
        created_at = v.get("created_at")
        if isinstance(created_at, datetime):
            if now - created_at > timedelta(minutes=SESSION_TTL_MINUTES):
                expired.append(k)
    for k in expired:
        SESSIONS.pop(k, None)

# ============================================================
# Allowed areas + damage vocab (KEEPING YOUR FORMAT)
# ============================================================

ALLOWED_AREAS = [
    "front bumper upper", "front bumper lower",
    "rear bumper upper", "rear bumper lower",
    "front left fender", "front right fender",
    "rear left fender", "rear right fender",
    "front left door", "front right door",
    "rear left door", "rear right door",
    "left quarter panel", "right quarter panel",
    "hood", "roof", "trunk", "tailgate",
    "windshield", "rear window",
    "left windows", "right windows",
    "left side mirror", "right side mirror",
    "left headlight", "right headlight",
    "left taillight", "right taillight",
    "left front wheel", "right front wheel",
    "left rear wheel", "right rear wheel",
    "left front tire", "right front tire",
    "left rear tire", "right rear tire",
]

ALLOWED_DAMAGE_TYPES = [
    "light scratch",
    "deep scratch",
    "paint scuff",
    "paint transfer",
    "small dent",
    "deep dent",
    "dent",
    "dented",
    "heavy dent",
    "heavily dented",
    "crease",
    "panel deformation",
    "bumper deformation",
    "plastic tear",
    "crack",
    "hole",
    "chip",
    "glass chip",
    "glass crack",
    "curb rash",
    "bent wheel",
    "misalignment",
    "deformed",
    "heavily deformed",
]


def clean_areas_from_text(text: str) -> List[str]:
    t = text.lower()
    found = [a for a in ALLOWED_AREAS if a in t]
    out: List[str] = []
    seen = set()
    for a in found:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def clean_damage_types_from_text(text: str) -> List[str]:
    t = text.lower()
    found = [d for d in ALLOWED_DAMAGE_TYPES if d in t]
    out: List[str] = []
    seen = set()
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}

# ============================================================
# Media download helpers
# ============================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")


def download_twilio_media(url: str) -> bytes:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio credentials not set")
    resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    resp.raise_for_status()
    return resp.content


def bytes_to_data_url(data: bytes, ctype: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{ctype};base64,{b64}"

# ============================================================
# PRE-SCAN PROMPT with LEFT/RIGHT FIX + ORIENTATION
# ============================================================

PRE_SCAN_SYSTEM_PROMPT = """
You are an automotive DAMAGE PRE-SCAN AI.

Your job:
- Look at the photo(s) of a vehicle.
- Identify ONLY the panels/areas that clearly show visible damage.
- Identify ONLY basic damage types.
- DO NOT guess or invent areas that are not clearly damaged.

LEFT/RIGHT RULE (IMPORTANT):
- Always use the DRIVER'S PERSPECTIVE for left/right (sitting inside the car facing forward).
- NEVER flip orientation based on camera viewing angle.
- If the angle makes it hard to be 100% sure, you can say “front-left (likely)” or “front-right (likely)” in NOTES,
  but the AREAS list must still use the correct left/right labels.
- DO NOT label damage as right-side if it is actually on the left, or vice versa.

ORIENTATION STEP:
Before listing damage, briefly describe the photo angle:
Examples:
- “Photo appears taken from the front-left of the vehicle.”
- “Photo taken straight from the front.”
- “Photo appears taken from the front-right.”

FORMAT TO USE:

ORIENTATION:
- one sentence about camera angle

AREAS:
- front left fender
- front bumper upper
- right headlight

DAMAGE TYPES:
- deep dent
- dent
- panel deformation
- crack

NOTES:
- short comments about what you see, plus any possible suspension/structural concerns
""".strip()

# ============================================================
# Run PRE-SCAN
# ============================================================

def run_pre_scan(image_data_urls: List[str], shop: Shop) -> Dict[str, Any]:
    if not image_data_urls:
        return {"areas": [], "damage_types": [], "notes": "", "raw": ""}

    content: List[dict] = [
        {
            "type": "text",
            "text": f"Customer photos for {shop.name}. Follow the PRE-SCAN instructions exactly.",
        }
    ]

    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": PRE_SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    raw = completion.choices[0].message.content or ""

    areas = clean_areas_from_text(raw)
    damage_types = clean_damage_types_from_text(raw)

    notes = ""
    m = re.search(r"NOTES:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        notes = m.group(1).strip()

    return {
        "areas": areas,
        "damage_types": damage_types,
        "notes": notes,
        "raw": raw,
    }


def build_pre_scan_sms(shop: Shop, pre: Dict[str, Any]) -> str:
    areas = pre.get("areas", [])
    damage_types = pre.get("damage_types", [])
    notes = pre.get("notes", "") or ""

    msg: List[str] = [f"AI Pre-Scan for {shop.name}", ""]

    if areas:
        msg.append("Here's what I can clearly see from your photo(s):")
        for a in areas:
            msg.append(f"- {a}")
    else:
        msg.append(
            "I couldn't confidently pick out specific damaged panels yet from this angle."
        )

    if damage_types:
        msg.append("")
        msg.append("Damage types I can see:")
        msg.append("- " + ", ".join(damage_types))

    if notes:
        msg.append("")
        msg.append("Notes:")
        msg.append(notes)

    msg.append("")
    msg.append("If this looks roughly correct, reply 1 and I'll send a full estimate with cost.")
    msg.append("If it's off, reply 2 and you can send clearer / wider photos.")

    return "\n".join(msg)

# ============================================================
# ESTIMATE SYSTEM PROMPT
# ============================================================

ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, Canada (year 2025).

You receive:
- CONFIRMED damaged areas (only from the photo-based pre-scan).
- CONFIRMED basic damage types.
- Optional notes (for example: “hood is heavily dented and deformed on the left side.”).

Your tasks:
1) Classify severity: "minor", "moderate", "severe", or "unknown".
2) Provide a realistic repair cost range in CAD (min_cost, max_cost) for labour + materials.
3) Write a short, customer-friendly explanation (2–4 sentences).
4) Include a brief disclaimer that this is a visual preliminary estimate only.

Severity guidance:
- Crushed, heavily dented, caved-in, or clearly deformed panels on structural areas (trunk, hood, bumpers, quarter panels, roof)
  are usually "moderate" or "severe", not "minor".
- Small cosmetic scratches or scuffs on a single panel with no deformation are usually "minor".
- Multiple damaged panels or combined deformation + cracks typically mean "moderate" or "severe".

STRICT RULES:
- NEVER add new panels or areas beyond those given.
- If confirmed areas list is empty, set severity to "unknown" and keep cost very low or 0–200.

Output JSON ONLY in this exact schema:

{
  "severity": "minor | moderate | severe | unknown",
  "min_cost": 0,
  "max_cost": 0,
  "summary": "2–4 sentence explanation.",
  "disclaimer": "Short note reminding it's a visual estimate only."
}
""".strip()

# ============================================================
# Run ESTIMATE + sanity adjustments
# ============================================================

def run_estimate(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    payload = {
        "shop_name": shop.name,
        "region": "Ontario",
        "year": 2025,
        "confirmed_areas": areas,
        "damage_types": damage_types,
        "pre_scan_notes": notes or "",
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                ],
            },
        ],
    )

    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)

    severity = (data.get("severity") or "unknown").lower()
    if severity not in {"minor", "moderate", "severe", "unknown"}:
        severity = "unknown"

    def _float(val, default=0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    estimate = {
        "severity": severity,
        "min_cost": _float(data.get("min_cost", 0)),
        "max_cost": _float(data.get("max_cost", 0)),
        "summary": str(data.get("summary", "")).strip(),
        "disclaimer": str(data.get("disclaimer", "")).strip(),
    }

    return sanity_adjust_estimate(estimate, areas, damage_types, notes)


def sanity_adjust_estimate(
    estimate: Dict[str, Any],
    areas: List[str],
    damage_types: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    severity = estimate.get("severity", "unknown").lower()
    def _float(val, default=0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    min_cost = _float(estimate.get("min_cost", 0))
    max_cost = _float(estimate.get("max_cost", 0))

    lowered_areas = [a.lower() for a in areas]
    lowered_types = [d.lower() for d in damage_types]
    notes_text = (notes or "").lower()

    wheel_like = [a for a in lowered_areas if "wheel" in a or "rim" in a]
    tire_like = [a for a in lowered_areas if "tire" in a]
    non_wheel_panels = [a for a in lowered_areas if a not in wheel_like + tire_like]

    # 1) Wheel-only jobs
    if non_wheel_panels == [] and (wheel_like or tire_like):
        serious = any(
            key in " ".join(lowered_types)
            for key in ["bent wheel", "crack", "deep dent", "hole", "puncture"]
        )
        if serious:
            min_cost = max(min_cost, 250)
            max_cost = max(max_cost, 700)
            severity = "moderate"
        else:
            min_cost = max(min_cost, 120)
            max_cost = max(max_cost, 450)
            severity = "minor"

    # 2) Heavy deformation overrides
    severity_rank = {"unknown": 0, "minor": 1, "moderate": 2, "severe": 3}
    rank_to_label = {v: k for k, v in severity_rank.items()}
    current_rank = severity_rank.get(severity, 0)

    heavy_terms = [
        "heavily dented",
        "heavy dent",
        "deep dent",
        "panel deformation",
        "bumper deformation",
        "deformed",
        "heavily deformed",
        "crushed",
        "caved in",
        "buckled",
        "pushed in",
        "folded",
    ]

    types_text = " ".join(lowered_types)
    has_heavy = any(term in types_text or term in notes_text for term in heavy_terms)

    structural_panels = [
        "hood",
        "trunk",
        "roof",
        "left quarter panel",
        "right quarter panel",
        "front bumper upper",
        "front bumper lower",
        "rear bumper upper",
        "rear bumper lower",
    ]
    has_structural = any(p in lowered_areas for p in structural_panels)

    if has_heavy and has_structural:
        current_rank = max(current_rank, severity_rank["severe"])
    elif has_heavy:
        current_rank = max(current_rank, severity_rank["moderate"])

    # 3) Multi-panel heuristic
    if len(non_wheel_panels) >= 3:
        current_rank = max(current_rank, severity_rank["moderate"])

    severity = rank_to_label.get(current_rank, severity)

    # 4) Guardrails on cost
    if severity == "minor":
        if min_cost <= 0:
            min_cost = 150
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 600
        max_cost = min(max_cost, 2000)
    elif severity == "moderate":
        if min_cost <= 0:
            min_cost = 600
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 2500
    elif severity == "severe":
        if min_cost <= 0:
            min_cost = 2000
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 6000
    else:  # unknown
        if min_cost <= 0 and max_cost <= 0:
            min_cost, max_cost = 0, 200

    if max_cost < min_cost:
        max_cost = min_cost

    # Round to nearest $10
    min_cost = int(round(min_cost / 10.0)) * 10
    max_cost = int(round(max_cost / 10.0)) * 10

    estimate["severity"] = severity
    estimate["min_cost"] = min_cost
    estimate["max_cost"] = max_cost
    return estimate


def build_estimate_sms(
    shop: Shop,
    areas: List[str],
    damage_types: List[str],
    estimate: Dict[str, Any],
) -> str:
    severity = estimate.get("severity", "unknown").capitalize()
    min_cost = int(estimate.get("min_cost", 0) or 0)
    max_cost = int(estimate.get("max_cost", 0) or 0)
    summary = estimate.get("summary") or ""
    disclaimer = estimate.get("disclaimer") or (
        "This is a visual pre-estimate only. Final pricing may change after in-person inspection."
    )

    estimate_id = str(uuid.uuid4())
    areas_str = ", ".join(areas) if areas else "not clearly identified yet"
    dmg_str = ", ".join(damage_types) if damage_types else "not clearly classified yet"

    lines: List[str] = [
        f"AI Damage Estimate for {shop.name}",
        "",
        f"Severity: {severity}",
        f"Estimated Cost (Ontario 2025): ${min_cost:,} – ${max_cost:,}",
        f"Areas: {areas_str}",
        f"Damage Types: {dmg_str}",
    ]

    if summary:
        lines.append("")
        lines.append(summary)

    lines.append("")
    lines.append(f"Estimate ID (internal): {estimate_id}")
    lines.append("")
    lines.append(disclaimer)

    return "\n".join(lines)

# ============================================================
# Twilio webhook (2-step flow)
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token) if token else None

    reply = MessagingResponse()

    if not shop:
        reply.message(
            "This number is not configured correctly. Please contact the body shop directly."
        )
        return Response(content=str(reply), media_type="application/xml")

    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From", "")
        num_media_str = form.get("NumMedia", "0") or "0"

        try:
            num_media = int(num_media_str)
        except ValueError:
            num_media = 0

        cleanup_sessions()
        key = session_key(shop, from_number)
        lower_body = body.lower()

        # CASE 1 – MMS received: run PRE-SCAN
        if num_media > 0:
            image_data_urls: List[str] = []

            for i in range(num_media):
                media_url = form.get(f"MediaUrl{i}")
                content_type = form.get(f"MediaContentType{i}") or "image/jpeg"
                if not media_url:
                    continue

                try:
                    raw = download_twilio_media(media_url)
                except Exception:
                    reply.message(
                        "Sorry — I couldn't download the photos from your carrier. "
                        "Please resend 1–3 clear photos of the damaged area."
                    )
                    return Response(content=str(reply), media_type="application/xml")

                data_url = bytes_to_data_url(raw, content_type)
                image_data_urls.append(data_url)

            if not image_data_urls:
                reply.message(
                    "Sorry — I had trouble reading the photos. "
                    "Please resend 1–3 clear photos of the damaged area."
                )
            else:
                try:
                    pre_scan = run_pre_scan(image_data_urls, shop)
                except Exception:
                    reply.message(
                        "Sorry — I couldn't process the photos. "
                        "Please try sending them again."
                    )
                    return Response(content=str(reply), media_type="application/xml")

                SESSIONS[key] = {
                    "areas": pre_scan.get("areas", []),
                    "damage_types": pre_scan.get("damage_types", []),
                    "notes": pre_scan.get("notes", ""),
                    "raw": pre_scan.get("raw", ""),
                    "created_at": datetime.utcnow(),
                }

                reply.message(build_pre_scan_sms(shop, pre_scan))

            return Response(content=str(reply), media_type="application/xml")

        # CASE 2 – Customer replies 1 (confirm pre-scan)
        if lower_body in {"1", "yes", "y"}:
            session = SESSIONS.get(key)
            if not session:
                reply.message(
                    "I don't see a recent photo for this number. "
                    "Please send 1–3 clear photos of the damaged area to start a new estimate."
                )
                return Response(content=str(reply), media_type="application/xml")

            try:
                estimate = run_estimate(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    session.get("notes", ""),
                )
                text = build_estimate_sms(
                    shop,
                    session.get("areas", []),
                    session.get("damage_types", []),
                    estimate,
                )
            except Exception:
                text = (
                    "Sorry — I couldn’t generate the estimate. "
                    "Please resend the photos to start again."
                )

            SESSIONS.pop(key, None)
            reply.message(text)
            return Response(content=str(reply), media_type="application/xml")

        # CASE 3 – Customer replies 2 (pre-scan is wrong)
        if lower_body in {"2", "no", "n"}:
            SESSIONS.pop(key, None)
            reply.message(
                "No problem. Please send 1–3 clearer photos of the damaged area "
                "(include a wide shot plus a couple close-ups) and I'll rescan it."
            )
            return Response(content=str(reply), media_type="application/xml")

        # CASE 4 – Default instructions
        instructions = (
            f"Hi from {shop.name}!\n\n"
            "To get an AI-powered damage estimate:\n"
            "1) Send 1–3 clear photos of the damaged area.\n"
            "2) I’ll send a quick Pre-Scan.\n"
            "3) Reply 1 if it looks right, or 2 if it’s off.\n"
            "4) Then I'll send your full Ontario 2025 cost estimate.\n\n"
            "You can start by sending photos now."
        )
        reply.message(instructions)
        return Response(content=str(reply), media_type="application/xml")

    except Exception:
        reply.message("Unexpected error. Please send the photos again.")
        return Response(content=str(reply), media_type="application/xml")

# ============================================================
# Admin: list shops + example webhook URLs
# ============================================================

@app.get("/admin/shops")
async def list_shops():
    base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    result = []
    for shop in SHOPS_BY_TOKEN.values():
        webhook = (
            f"{base_url}/sms-webhook?token={shop.webhook_token}"
            if base_url
            else f"/sms-webhook?token={shop.webhook_token}"
        )
        result.append(
            {
                "id": shop.id,
                "name": shop.name,
                "webhook_token": shop.webhook_token,
                "twilio_webhook_example": webhook,
            }
        )
    return result

# ============================================================
# Health check
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-shop AI damage estimator is running"}
