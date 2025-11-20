import os
import json
from typing import Dict, List, Tuple

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI

# ------------
# App & client
# ------------

app = FastAPI()
client = OpenAI()  # uses OPENAI_API_KEY from env


# -----------------------
# Shop configuration
# -----------------------

def load_shops() -> Dict[str, dict]:
    """
    Load shops from SHOPS_JSON env var.

    SHOPS_JSON example:
    [
      {"id": "shop1", "name": "Brampton Auto Body", "webhook_token": "brampton123"},
      {"id": "shop2", "name": "Mississauga Collision Centre", "webhook_token": "miss_centre_456"}
    ]
    """
    raw = os.getenv("SHOPS_JSON", "[]")
    try:
        data = json.loads(raw)
        shops_by_token = {s["webhook_token"]: s for s in data if "webhook_token" in s}
        return shops_by_token
    except Exception:
        return {}


SHOPS_BY_TOKEN = load_shops()
DEFAULT_SHOP_NAME = "Mississauga Collision Centre"

# -----------------------
# Damage area whitelist
# -----------------------

ALLOWED_AREAS = [
    "front bumper upper", "front bumper lower", "rear bumper upper", "rear bumper lower",
    "front left fender", "front right fender", "rear left fender", "rear right fender",
    "front left door", "front right door", "rear left door", "rear right door",
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
    "rocker panel / sill", "left rocker panel / sill", "right rocker panel / sill",
]


def clean_area_list(text: str) -> List[str]:
    """Return the subset of ALLOWED_AREAS that actually appear in the model text."""
    text_lower = text.lower()
    final: List[str] = []
    for area in ALLOWED_AREAS:
        if area in text_lower:
            final.append(area)
    # remove duplicates while preserving order
    seen = set()
    unique: List[str] = []
    for a in final:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


# -----------------------
# Prompts
# -----------------------

PRE_SCAN_PROMPT = """
You are an automotive AI DAMAGE PRE-SCAN system for collision photos.

Your job:
- Look ONLY at the visible damage in the provided photo.
- Identify WHERE the damage is on the vehicle using ONLY this allowed list of areas:
  front bumper upper, front bumper lower, rear bumper upper, rear bumper lower,
  front left fender, front right fender, rear left fender, rear right fender,
  front left door, front right door, rear left door, rear right door,
  left quarter panel, right quarter panel,
  hood, roof, trunk, tailgate,
  windshield, rear window, left windows, right windows,
  left side mirror, right side mirror,
  left headlight, right headlight,
  left taillight, right taillight,
  left front wheel, right front wheel,
  left rear wheel, right rear wheel,
  left front tire, right front tire,
  left rear tire, right rear tire,
  rocker panel / sill, left rocker panel / sill, right rocker panel / sill.

Rules:
1. ONLY list areas where damage is clearly visible in the photo.
2. NEVER mention an area if you cannot clearly see damage there.
3. If you truly cannot see any damage, say "none".
4. Keep it very short and direct.

Output format EXACTLY:

AREAS:
- area 1
- area 2

NOTES:
- one short sentence about what you see overall
"""

FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator for Ontario, Canada (year 2025).
You are preparing an SMS-style estimate for a body shop customer.

Rules:
1. ONLY use the confirmed areas the user provides. Do NOT introduce new areas.
2. Be realistic and conservative with severity and cost – like an experienced estimator.
3. Assume mid-range pricing for a reputable collision centre in Ontario (not the cheapest, not the most expensive).
4. Consider typical body work: dent repair, panel replacement, paint, blending, wheel refinishing, glass, sensors, etc.
5. Answer in clear, friendly language suitable for a customer text message.

Structure your reply EXACTLY like this:

AI Damage Estimate for {shop_name}

Severity: <Mild / Moderate / Severe>
Estimated Cost (Ontario 2025): $x – $y

Areas:
- area 1
- area 2

Damage Types:
- short bullet 1
- short bullet 2

Explanation:
Short 3–4 sentence explanation in plain language.
"""

# -----------------------
# Simple in-memory session store
# -----------------------

# key = phone number, value = dict(areas=[...], shop_name=str)
sessions: Dict[str, Dict[str, object]] = {}


def get_shop_name(token: str | None) -> str:
    if token and token in SHOPS_BY_TOKEN:
        return SHOPS_BY_TOKEN[token].get("name", DEFAULT_SHOP_NAME)
    return DEFAULT_SHOP_NAME


# -----------------------
# Helpers
# -----------------------

def parse_pre_scan_output(text: str) -> Tuple[List[str], str]:
    """
    Parse the PRE-SCAN model output into (areas, notes_text).
    Falls back gracefully if the format isn't perfect.
    """
    areas: List[str] = []
    notes = ""

    # split on "NOTES:"
    parts = text.split("NOTES:")
    areas_block = parts[0]
    if len(parts) > 1:
        notes_block = parts[1].strip()
        # keep just first line or bullet for SMS brevity
        notes_lines = [ln.strip() for ln in notes_block.splitlines() if ln.strip()]
        if notes_lines:
            notes = notes_lines[0]

    for line in areas_block.splitlines():
        line = line.strip()
        if line.startswith("-"):
            areas.append(line.lstrip("-").strip())

    # Run through whitelist
    cleaned = clean_area_list("\n".join(areas))

    return cleaned, notes


# -----------------------
# Routes
# -----------------------

@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    # Twilio sends application/x-www-form-urlencoded
    form = await request.form()
    body = (form.get("Body") or "").strip()
    lower_body = body.lower()
    from_number = form.get("From") or "unknown"
    media_url = form.get("MediaUrl0")  # first image only for now

    token = request.query_params.get("token")
    shop_name = get_shop_name(token)

    # Start or help message
    if lower_body in {"hi", "hello", "start"} and not media_url:
        msg = (
            f"Hi! This is the AI damage estimator for {shop_name}.\n\n"
            "📸 Please reply with 1–3 clear photos of the damage:\n"
            "- Step back so the whole damaged area is visible\n"
            "- Include the edge of nearby panels if possible\n\n"
            "After the AI Pre-Scan, reply 1 to confirm the areas or 2 to resend clearer photos."
        )
        return PlainTextResponse(msg)

    # Step 1: image received → Pre-scan
    if media_url:
        try:
            pre_scan = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PRE_SCAN_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this photo for visible vehicle damage."
                            },
                            {"type": "image_url", "image_url": {"url": media_url}},
                        ],
                    },
                ],
            )
            raw_text = pre_scan.choices[0].message.content or ""
        except Exception:
            # Fail gracefully for the customer
            return PlainTextResponse(
                "Sorry, our AI pre-scan is temporarily unavailable. "
                "Please call the shop directly to discuss your damage."
            )

        areas, notes = parse_pre_scan_output(raw_text)

        # Store session for this number
        sessions[from_number] = {"areas": areas, "shop_name": shop_name}

        if not areas:
            reply = (
                f"AI Pre-Scan for {shop_name}\n\n"
                "I couldn't clearly see damage from this photo.\n"
                "Please try again with a closer photo of the damaged area, "
                "making sure the lighting is good and the damage is in focus."
            )
            return PlainTextResponse(reply)

        reply_lines = [
            f"AI Pre-Scan for {shop_name}",
            "",
            "I can see damage in these areas:",
        ]
        for a in areas:
            reply_lines.append(f"- {a}")

        if notes:
            reply_lines.append("")
            reply_lines.append(f"Notes: {notes}")

        reply_lines.append("")
        reply_lines.append("Reply 1 if this list looks roughly correct.")
        reply_lines.append("Reply 2 if it's wrong and you'll resend clearer photos.")

        return PlainTextResponse("\n".join(reply_lines))

    # Step 2: user confirms areas with "1"
    if lower_body == "1":
        session = sessions.get(from_number)
        if not session:
            return PlainTextResponse(
                "I don't have a recent photo from you. "
                "Please send a damage photo first so I can run an AI Pre-Scan."
            )

        confirmed_areas: List[str] = session.get("areas", [])
        shop_name = str(session.get("shop_name", shop_name))

        try:
            estimate = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": FULL_ESTIMATE_PROMPT.format(shop_name=shop_name),
                    },
                    {
                        "role": "user",
                        "content": f"Confirmed damage areas: {', '.join(confirmed_areas)}",
                    },
                ],
            )
            estimate_text = estimate.choices[0].message.content or ""
        except Exception:
            return PlainTextResponse(
                "Sorry, I couldn't generate the full estimate right now. "
                "Please call the shop to continue your quote."
            )

        return PlainTextResponse(estimate_text)

    # Step 2: user says list was wrong
    if lower_body == "2":
        msg = (
            "No problem. Please resend 1–3 clearer photos of the damaged area:\n"
            "- Move a bit closer so the damage fills more of the frame\n"
            "- Avoid glare / strong reflections if possible\n"
            "- Make sure the image is not blurry\n\n"
            "I'll run a fresh AI Pre-Scan when I receive the new photo."
        )
        return PlainTextResponse(msg)

    # Fallback
    fallback = (
        f"This is the AI damage estimator for {shop_name}.\n\n"
        "To begin, please send a clear photo of your vehicle's damage, "
        "or reply 'hi' for instructions."
    )
    return PlainTextResponse(fallback)
