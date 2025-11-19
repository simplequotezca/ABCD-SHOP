import os
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# ------------- FastAPI / OpenAI setup -------------

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------- Shop config (from SHOPS_JSON) -------------

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops_from_env() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON environment variable is required")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SHOPS_JSON must be valid JSON") from exc

    shops_by_token: Dict[str, Shop] = {}
    for item in data:
        shop = Shop(**item)
        shops_by_token[shop.webhook_token] = shop
    return shops_by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops_from_env()


# ------------- In-memory state for 2-step flow -------------

# Key = f"{shop_id}:{from_number}"
PENDING_PRE_SCANS: Dict[str, Dict[str, Any]] = {}
PRE_SCAN_TTL_MINUTES = 60  # how long a pre-scan is valid


def conversation_key(shop_id: str, from_number: str) -> str:
    return f"{shop_id}:{from_number}"


def cleanup_expired_pre_scans() -> None:
    """Remove old pre-scans so memory doesn't grow forever."""
    now = datetime.utcnow()
    expired_keys: List[str] = []
    for key, value in PENDING_PRE_SCANS.items():
        created_at: datetime = value.get("created_at")  # type: ignore
        if isinstance(created_at, datetime):
            if now - created_at > timedelta(minutes=PRE_SCAN_TTL_MINUTES):
                expired_keys.append(key)

    for key in expired_keys:
        PENDING_PRE_SCANS.pop(key, None)


# ------------- Shared damage vocabulary -------------

ALLOWED_AREAS = [
    # Front
    "front bumper upper",
    "front bumper lower",
    "front grille",
    "hood",
    "left front fender",
    "right front fender",
    "left front door",
    "right front door",
    "left side mirror",
    "right side mirror",
    "front windshield",
    "front left headlight",
    "front right headlight",
    # Rear
    "rear bumper upper",
    "rear bumper lower",
    "trunk lid / tailgate",
    "rear left quarter panel",
    "rear right quarter panel",
    "rear windshield",
    "rear left taillight",
    "rear right taillight",
    # Roof & sides
    "roof",
    "roof rail / pillar",
    "left rear door",
    "right rear door",
    # Wheels / tires
    "front left wheel / rim",
    "front right wheel / rim",
    "rear left wheel / rim",
    "rear right wheel / rim",
    "front left tire",
    "front right tire",
    "rear left tire",
    "rear right tire",
]

ALLOWED_DAMAGE_TYPES = [
    "light scratch",
    "deep scratch",
    "paint scuff",
    "scuff / transfer",
    "small dent",
    "deep dent",
    "crease",
    "panel deformation",
    "bumper deformation",
    "plastic tear",
    "crack",
    "hole / puncture",
    "glass chip",
    "glass crack",
    "misalignment",
    "gap / panel misfit",
    "curb rash",
    "bent wheel",
]


# ------------- Helpers for OpenAI JSON parsing -------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Try hard to parse a JSON object out of the model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract a {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: empty structure
    return {}


# ------------- Step 1: Vision pre-scan (areas only) -------------

def call_pre_scan_model(image_url: str) -> Dict[str, Any]:
    """
    Ask the vision model ONLY for clearly visible areas + damage types.
    No prices, no guessing, must stick to ALLOWED_AREAS / ALLOWED_DAMAGE_TYPES.
    """
    system_prompt = f"""
You are an auto body damage PRE-SCAN assistant.

Your ONLY job:
- Look at the photo of a vehicle.
- List which panels/areas appear clearly damaged.
- List simple damage types.
- DO NOT estimate cost.
- DO NOT mention anything that is not clearly visible.
- DO NOT guess about hidden sides of the vehicle.

Very strict rules:
1. If you are not clearly sure an area is damaged, DO NOT include it.
2. If the image only shows the rear, NEVER mention front parts.
3. Use ONLY the areas from this list (verbatim strings):

{ALLOWED_AREAS}

4. Use ONLY the damage_types from this list (verbatim strings):

{ALLOWED_DAMAGE_TYPES}

5. If you are unsure about areas, return an empty list for "areas" and explain in "notes".

Respond with JSON ONLY in this exact schema:

{{
  "areas": ["area1", "area2"],
  "damage_types": ["type1", "type2"],
  "side": "front" | "rear" | "left" | "right" | "roof" | "multiple" | "unknown",
  "confidence": 0.0,
  "notes": "short note about what is visible / any uncertainty"
}}
"""

    user_content = [
        {
            "type": "text",
            "text": "Analyze this photo for VISIBLE vehicle damage only, following the JSON schema.",
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = completion.choices[0].message.content or ""
    data = safe_json_loads(raw)

    # Normalise + hard-filter against allowed vocab
    areas = [a for a in data.get("areas", []) if a in ALLOWED_AREAS]
    damage_types = [d for d in data.get("damage_types", []) if d in ALLOWED_DAMAGE_TYPES]

    side = (data.get("side") or "unknown").lower()
    if side not in {"front", "rear", "left", "right", "roof", "multiple", "unknown"}:
        side = "unknown"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    notes = str(data.get("notes", "")).strip()

    return {
        "areas": areas,
        "damage_types": damage_types,
        "side": side,
        "confidence": max(0.0, min(confidence, 1.0)),
        "notes": notes,
    }


def build_pre_scan_sms(shop: Shop, pre_scan: Dict[str, Any]) -> str:
    areas = pre_scan.get("areas", []) or []
    damage_types = pre_scan.get("damage_types", []) or []
    notes = pre_scan.get("notes") or ""
    side = pre_scan.get("side") or "unknown"

    header = f"AI Pre-Scan for {shop.name}"

    if not areas:
        body_lines = [
            "I couldn't clearly identify specific damaged panels from this photo.",
            "",
            "To improve accuracy, please send:",
            "- 1 wide shot of the whole damaged corner, and",
            "- 1–2 closer shots from different angles.",
        ]
    else:
        areas_str = ", ".join(areas)
        dmg_str = ", ".join(damage_types) if damage_types else "not clearly identified"

        side_label = ""
        if side != "unknown":
            side_label = f" (mainly {side} side)"

        body_lines = [
            f"I can see damage in these areas{side_label}:",
            f"- {areas_str}",
            "",
            f"Damage types: {dmg_str}.",
        ]
        if notes:
            body_lines.append(f"Notes: {notes}")

    confirm_lines = [
        "",
        "Reply 1 if this looks roughly correct.",
        "Reply 2 if it's wrong and you'll resend clearer photos.",
    ]

    return header + "\n\n" + "\n".join(body_lines + confirm_lines)


# ------------- Step 2: Cost estimate from pre-scan -------------

def call_estimate_model(pre_scan: Dict[str, Any], extra_info: str = "") -> Dict[str, Any]:
    """
    Take the pre-scan JSON and (optionally) extra text info (VIN, notes)
    and produce a severity + Ontario 2025 cost band.
    """
    system_prompt = """
You are an experienced auto body estimator in Ontario, Canada, using realistic 2025 prices.

Input:
- A JSON pre-scan listing VISIBLE damaged areas and damage types.
- Optional short text from the customer.

Your goals:
- Classify severity as "minor", "moderate", or "severe".
- Produce a conservative but realistic **labour + materials** cost range in CAD dollars.
- Keep it as a **ballpark estimate**, not a guarantee.

Rules:
1. NEVER invent damage to areas that are not in the pre-scan areas list.
2. If the pre-scan areas list is empty, say severity "unknown" and keep costs very low (e.g. 0–200) or recommend in-person inspection.
3. Consider that major structural or multiple-panel damage can be much higher.
4. Output only JSON with this schema:

{
  "severity": "minor" | "moderate" | "severe" | "unknown",
  "min_cost": 0,
  "max_cost": 0,
  "summary": "2–4 sentence explanation the shop can send to a customer.",
  "disclaimer": "Short sentence reminding them it's a visual estimate only."
}
"""

    user_text = {
        "pre_scan_json": pre_scan,
        "extra_info": extra_info or "",
        "region": "Ontario",
        "year": 2025,
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(user_text),
                    }
                ],
            },
        ],
    )

    raw = completion.choices[0].message.content or ""
    data = safe_json_loads(raw)

    severity = (data.get("severity") or "unknown").lower()
    if severity not in {"minor", "moderate", "severe", "unknown"}:
        severity = "unknown"

    try:
        min_cost = float(data.get("min_cost", 0) or 0)
    except (TypeError, ValueError):
        min_cost = 0.0

    try:
        max_cost = float(data.get("max_cost", 0) or 0)
    except (TypeError, ValueError):
        max_cost = 0.0

    summary = str(data.get("summary", "")).strip()
    disclaimer = str(data.get("disclaimer", "")).strip()

    estimate = {
        "severity": severity,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "summary": summary,
        "disclaimer": disclaimer,
    }

    return sanity_adjust_estimate(estimate, pre_scan)


def sanity_adjust_estimate(estimate: Dict[str, Any], pre_scan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply hard guardrails so we don't send obviously crazy numbers.
    Especially for rim-only jobs and basic damage.
    """
    severity = estimate.get("severity", "unknown").lower()
    try:
        min_cost = float(estimate.get("min_cost", 0) or 0)
    except (TypeError, ValueError):
        min_cost = 0.0

    try:
        max_cost = float(estimate.get("max_cost", 0) or 0)
    except (TypeError, ValueError):
        max_cost = 0.0

    areas = [a.lower() for a in pre_scan.get("areas", []) or []]
    damage_types = [d.lower() for d in pre_scan.get("damage_types", []) or []]

    wheel_like = [a for a in areas if "wheel" in a or "rim" in a]
    tire_like = [a for a in areas if "tire" in a]
    non_wheel_panels = [a for a in areas if a not in wheel_like + tire_like]

    # --- Special case: rim / wheel only jobs ---
    if non_wheel_panels == [] and (wheel_like or tire_like):
        # Light curb rash-type jobs
        serious_wheel = any(
            key in " ".join(damage_types)
            for key in ["bent wheel", "crack", "deep dent", "hole", "puncture"]
        )
        if serious_wheel:
            # Straightening / replacement
            min_cost = max(min_cost, 250)
            max_cost = max(max_cost, 600)
        else:
            # Simple curb rash refinish
            min_cost = 120
            max_cost = 450
        severity = "minor" if not serious_wheel else "moderate"

    # --- General severity guardrails ---
    if severity == "minor":
        if min_cost <= 0:
            min_cost = 150
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 600
        # Hard cap for minor
        max_cost = min(max_cost, 2000)

    elif severity == "moderate":
        if min_cost <= 0:
            min_cost = 600
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 2000

    elif severity == "severe":
        if min_cost <= 0:
            min_cost = 1800
        if max_cost <= 0 or max_cost < min_cost:
            max_cost = min_cost + 5000

    else:  # unknown
        if min_cost <= 0 and max_cost <= 0:
            min_cost, max_cost = 0, 200

    if max_cost < min_cost:
        max_cost = min_cost

    # Round to nearest $10 for nice SMS
    min_cost = int(round(min_cost / 10.0)) * 10
    max_cost = int(round(max_cost / 10.0)) * 10

    estimate["severity"] = severity
    estimate["min_cost"] = min_cost
    estimate["max_cost"] = max_cost
    return estimate


def build_estimate_sms(shop: Shop, pre_scan: Dict[str, Any], estimate: Dict[str, Any]) -> str:
    areas = pre_scan.get("areas", []) or []
    damage_types = pre_scan.get("damage_types", []) or []

    severity = estimate.get("severity", "unknown").capitalize()
    min_cost = int(estimate.get("min_cost", 0) or 0)
    max_cost = int(estimate.get("max_cost", 0) or 0)
    summary = estimate.get("summary") or ""
    disclaimer = estimate.get("disclaimer") or "This is a visual estimate only. Final pricing may change after in-person inspection."

    estimate_id = str(uuid.uuid4())

    areas_str = ", ".join(areas) if areas else "not clearly identified"
    dmg_str = ", ".join(damage_types) if damage_types else "not clearly identified"

    lines = [
        f"AI Damage Estimate for {shop.name}",
        "",
        f"Severity: {severity}",
        f"Estimated Cost (Ontario 2025): ${min_cost:,} – ${max_cost:,}",
        f"Areas: {areas_str}",
        f"Damage Types: {dmg_str}",
        "",
        summary,
        "",
        f"Estimate ID (internal): {estimate_id}",
        "",
        disclaimer,
    ]

    return "\n".join(lines)


# ------------- FastAPI Twilio webhook -------------

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Twilio will POST here:
    https://web-production-a1388.up.railway.app/sms-webhook?token=WEBHOOK_TOKEN
    """

    # Match shop by token
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        # Respond nicely so Twilio doesn't keep retrying
        resp = MessagingResponse()
        resp.message("Invalid shop configuration. Please contact the shop directly.")
        return Response(content=str(resp), media_type="application/xml")

    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From", "")
    num_media_str = form.get("NumMedia", "0") or "0"

    try:
        num_media = int(num_media_str)
    except ValueError:
        num_media = 0

    cleanup_expired_pre_scans()
    key = conversation_key(shop.id, from_number)

    resp = MessagingResponse()

    # --- CASE 1: Incoming photo(s) => run PRE-SCAN ---
    if num_media > 0:
        image_url = form.get("MediaUrl0")
        if not image_url:
            resp.message(
                "I couldn't read the photo. Please try again and make sure at least one image is attached."
            )
            return Response(content=str(resp), media_type="application/xml")

        pre_scan = call_pre_scan_model(image_url)

        # Store for step 2
        PENDING_PRE_SCANS[key] = {
            "shop_id": shop.id,
            "from_number": from_number,
            "pre_scan": pre_scan,
            "created_at": datetime.utcnow(),
        }

        sms_text = build_pre_scan_sms(shop, pre_scan)
        resp.message(sms_text)
        return Response(content=str(resp), media_type="application/xml")

    # --- CASE 2: No photo, but user replies "1" (confirm) or "2" (reject) ---

    lower_body = body.lower()

    if lower_body in {"1", "yes", "y"}:
        pending = PENDING_PRE_SCANS.get(key)
        if not pending:
            resp.message(
                "I don't see a recent photo for your number. "
                "Please send 2–3 clear photos of the damage to start a new estimate."
            )
            return Response(content=str(resp), media_type="application/xml")

        pre_scan = pending["pre_scan"]
        estimate = call_estimate_model(pre_scan, extra_info="")
        sms_text = build_estimate_sms(shop, pre_scan, estimate)

        # Clear state now that we've completed the estimate
        PENDING_PRE_SCANS.pop(key, None)

        resp.message(sms_text)
        return Response(content=str(resp), media_type="application/xml")

    if lower_body in {"2", "no", "n"}:
        # User says pre-scan was wrong
        PENDING_PRE_SCANS.pop(key, None)
        resp.message(
            "No problem. Please send 2–3 clearer photos of the damage "
            "(wide shot of the corner plus a couple of close-ups) and I'll rescan it."
        )
        return Response(content=str(resp), media_type="application/xml")

    # --- CASE 3: No media, no 1/2 => general instructions / fallback ---

    instructions = (
        f"Thanks for contacting {shop.name}.\n\n"
        "To get an AI damage estimate:\n"
        "1) Send 2–3 clear photos of the damaged area (different angles).\n"
        "2) I'll first send you a PRE-SCAN listing which panels look damaged.\n"
        "3) Reply 1 to confirm, or 2 if it's wrong and you'll resend photos.\n"
        "4) After you confirm, I'll send a detailed Ontario 2025 cost estimate.\n"
    )
    resp.message(instructions)
    return Response(content=str(resp), media_type="application/xml")


# ------------- Simple health check -------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI damage estimator is running"}
