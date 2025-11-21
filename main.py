import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
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
# Multi-shop config via SHOPS_JSON
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        default = Shop(
            id="default",
            name="Auto Body Shop",
            webhook_token="demo_token",
        )
        return {default.webhook_token: default}

    data = json.loads(raw)
    by_token = {}
    for item in data:
        shop = Shop(**item)
        by_token[shop.webhook_token] = shop
    return by_token


SHOPS_BY_TOKEN = load_shops()

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
    for key, data in SESSIONS.items():
        created = data.get("created_at")
        if isinstance(created, datetime):
            if now - created > timedelta(minutes=SESSION_TTL_MINUTES):
                expired.append(key)
    for k in expired:
        SESSIONS.pop(k, None)

# ============================================================
# Allowed vocabulary (keeps AI predictable)
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
    "windshield", "rear window", "left windows", "right windows",
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
]


def clean_areas_from_text(text: str) -> List[str]:
    t = text.lower()
    return list(dict.fromkeys([a for a in ALLOWED_AREAS if a in t]))


def clean_damage_types_from_text(text: str) -> List[str]:
    t = text.lower()
    return list(dict.fromkeys([d for d in ALLOWED_DAMAGE_TYPES if d in t]))


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
# PRE-SCAN (Areas + basic damage)
# ============================================================

PRE_SCAN_SYSTEM_PROMPT = """
You are an automotive DAMAGE PRE-SCAN AI.

Your job:
- Look at the image(s) of a vehicle.
- Identify ONLY the panels/areas that clearly show visible damage.
- Identify ONLY basic damage types.
- Do NOT guess. Keep it short.
- No cost estimates.

Format:

AREAS:
- area 1
- area 2

DAMAGE TYPES:
- type 1
- type 2

NOTES:
- any uncertainty
""".strip()


def run_pre_scan(image_urls: List[str], shop: Shop) -> Dict[str, Any]:
    content = [
        {
            "type": "text",
            "text": (
                f"Customer photos for {shop.name}. "
                "Analyze ONLY visible damage according to the PRE-SCAN rules."
            ),
        }
    ]

    for url in image_urls[:3]:
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
    types = clean_damage_types_from_text(raw)

    notes = ""
    m = re.search(r"NOTES:\s*(.+)", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        notes = m.group(1).strip()

    return {
        "areas": areas,
        "damage_types": types,
        "notes": notes,
        "raw": raw,
    }


def build_pre_scan_sms(shop: Shop, pre: Dict[str, Any]) -> str:
    lines = [f"AI Pre-Scan for {shop.name}", ""]

    if pre["areas"]:
        lines.append("Here's what I can clearly see:")
        for a in pre["areas"]:
            lines.append(f"- {a}")
    else:
        lines.append("I couldn't confidently see specific damaged panels yet.")

    if pre["damage_types"]:
        lines.append("")
        lines.append("Damage types:")
        lines.append("- " + ", ".join(pre["damage_types"]))

    if pre["notes"]:
        lines.append("")
        lines.append("Notes:")
        lines.append(pre["notes"])

    lines.append("")
    lines.append("If this looks right, reply 1.")
    lines.append("If it's off, reply 2 and send 1–3 clearer photos.")

    return "\n".join(lines)

# ============================================================
# FULL ESTIMATE
# ============================================================

ESTIMATE_SYSTEM_PROMPT = """
You are an experienced collision estimator in Ontario, Canada (2025).

You will receive:
- confirmed damaged areas
- damage types

Tasks:
1) Classify severity (minor, moderate, severe, unknown)
2) Provide CAD cost range
3) Provide a 2–4 sentence explanation
4) Provide a disclaimer

STRICT:
- Never add new areas.

Return ONLY JSON:
{
  "severity": "...",
  "min_cost": 0,
  "max_cost": 0,
  "summary": "",
  "disclaimer": ""
}
""".strip()


def run_estimate(shop: Shop, areas: List[str], dmg_types: List[str]) -> Dict[str, Any]:
    payload = {
        "shop_name": shop.name,
        "region": "Ontario",
        "year": 2025,
        "confirmed_areas": areas,
        "damage_types": dmg_types,
    }

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ESTIMATE_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    )

    raw = completion.choices[0].message.content or "{}"
    data = safe_json_loads(raw)

    severity = (data.get("severity") or "unknown").lower()
    if severity not in {"minor", "moderate", "severe", "unknown"}:
        severity = "unknown"

    try:
        min_cost = float(data.get("min_cost") or 0)
        max_cost = float(data.get("max_cost") or 0)
    except:
        min_cost, max_cost = 0, 0

    summary = data.get("summary", "").strip()
    disclaimer = data.get("disclaimer", "").strip()

    estimate = {
        "severity": severity,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "summary": summary,
        "disclaimer": disclaimer,
    }

    return estimate


def build_estimate_sms(shop: Shop, areas, dmg_types, estimate) -> str:
    return (
        f"AI Damage Estimate for {shop.name}\n\n"
        f"Severity: {estimate['severity'].capitalize()}\n"
        f"Estimated Cost (Ontario 2025): ${int(estimate['min_cost']):,} – ${int(estimate['max_cost']):,}\n"
        f"Areas: {', '.join(areas) if areas else 'None detected'}\n"
        f"Damage Types: {', '.join(dmg_types) if dmg_types else 'None detected'}\n\n"
        f"{estimate['summary']}\n\n"
        f"Estimate ID: {uuid.uuid4()}\n\n"
        f"{estimate['disclaimer'] or 'This is a visual estimate only.'}"
    )

# ============================================================
# Twilio Webhook (SAFE VERSION)
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token) if token else None
    reply = MessagingResponse()

    if not shop:
        reply.message("This number is not configured correctly.")
        return Response(str(reply), media_type="application/xml")

    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From", "")
        lower = body.lower()

        num_media = int(form.get("NumMedia", "0") or "0")

        cleanup_sessions()
        key = session_key(shop, from_number)

        # ----------------------------------------------------
        # CASE 1 — MEDIA RECEIVED
        # ----------------------------------------------------
        if num_media > 0:
            image_urls = []
            for i in range(num_media):
                url = form.get(f"MediaUrl{i}")
                if url:
                    image_urls.append(url)

            try:
                pre = run_pre_scan(image_urls, shop)
            except:
                reply.message("Sorry — I couldn't process the photos. Please try again.")
                return Response(str(reply), media_type="application/xml")

            SESSIONS[key] = {
                "areas": pre["areas"],
                "damage_types": pre["damage_types"],
                "created_at": datetime.utcnow(),
            }

            reply.message(build_pre_scan_sms(shop, pre))
            return Response(str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # CASE 2 — CONFIRM (1)
        # ----------------------------------------------------
        if lower in {"1", "yes", "y"}:
            session = SESSIONS.get(key)
            if not session:
                reply.message("I don't see recent photos. Send 1–3 photos to start.")
                return Response(str(reply), media_type="application/xml")

            try:
                estimate = run_estimate(shop, session["areas"], session["damage_types"])
                text = build_estimate_sms(shop, session["areas"], session["damage_types"], estimate)
            except:
                text = "Sorry — I couldn't generate the estimate. Please resend the photos."

            SESSIONS.pop(key, None)
            reply.message(text)
            return Response(str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # CASE 3 — RESCAN (2)
        # ----------------------------------------------------
        if lower in {"2", "no", "n"}:
            SESSIONS.pop(key, None)
            reply.message("No problem. Please send 1–3 clearer photos of the damaged area.")
            return Response(str(reply), media_type="application/xml")

        # ----------------------------------------------------
        # DEFAULT: Instructions
        # ----------------------------------------------------
        instructions = (
            f"Hi from {shop.name}!\n\n"
            "To get an AI-powered damage estimate:\n"
            "1) Send 1–3 clear photos of the damaged area.\n"
            "2) I’ll send a quick Pre-Scan.\n"
            "3) Reply 1 if it looks right, or 2 if it’s off.\n"
            "4) Then I'll send your full Ontario 2025 cost estimate.\n"
        )

        reply.message(instructions)
        return Response(str(reply), media_type="application/xml")

    except Exception:
        reply.message("Unexpected error. Please send the photos again.")
        return Response(str(reply), media_type="application/xml")

# ============================================================
# Admin helper
# ============================================================

@app.get("/admin/shops")
async def list_shops():
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    items = []
    for shop in SHOPS_BY_TOKEN.values():
        webhook = f"{base}/sms-webhook?token={shop.webhook_token}" if base else f"/sms-webhook?token={shop.webhook_token}"
        items.append(
            {"id": shop.id, "name": shop.name, "webhook_token": shop.webhook_token, "webhook": webhook}
        )
    return items

# ============================================================
# Health check
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-shop damage estimator running"}
