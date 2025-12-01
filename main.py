import os
import json
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI

# ============================================================
# ENV + OPENAI CLIENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()


# ============================================================
# SHOP CONFIG (MULTI-SHOP VIA TOKEN)
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str  # used in ?token=...
    # you can add calendar_id, pricing, etc. later if needed


def load_shops() -> Dict[str, Shop]:
    """
    SHOPS_JSON example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123"
      }
    ]
    """
    if not SHOPS_JSON:
        raise RuntimeError(
            "SHOPS_JSON env var is required. "
            "It must be a JSON array of shops."
        )

    try:
        raw = json.loads(SHOPS_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid SHOPS_JSON: {e}")

    by_token: Dict[str, Shop] = {}
    for item in raw:
        shop = Shop(**item)
        by_token[shop.webhook_token] = shop
    return by_token


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()


def get_shop_by_token(token: str) -> Optional[Shop]:
    return SHOPS_BY_TOKEN.get(token)


# ============================================================
# AI DAMAGE ESTIMATOR (VISION)
# ============================================================

def build_estimator_prompt(shop: Shop) -> str:
    return (
        f"You are an expert auto body damage estimator working for {shop.name} "
        f"in Ontario, Canada in 2025.\n\n"
        "You will receive 1–3 photos of collision damage.\n"
        "- Carefully inspect all visible panels, lights, bumpers, trunk/hood, doors, etc.\n"
        "- Identify which areas are damaged and the TYPE of damage "
        "(scratches, dents, deep dents, cracks, panel deformation, misalignment, etc.).\n"
        "- Estimate SEVERITY as one of: Minor, Moderate, Severe.\n"
        "- Provide a realistic estimated cost range in CAD based on typical Ontario body shop pricing.\n"
        "- Mention important repair steps (e.g., remove & replace bumper cover, repair & refinish, "
        "blend adjacent panels, check sensors, wheel alignment, frame check if needed).\n"
        "- Do NOT promise final pricing. Clearly say this is a visual PRELIMINARY estimate only.\n"
    )


def analyze_damage_with_vision(image_urls: List[str], shop: Shop) -> str:
    """
    Uses OpenAI chat completions with vision support.

    IMPORTANT: we use `image_url` (not `input_image`) so it works with your current OpenAI setup.
    """
    if not image_urls:
        raise ValueError("No image URLs provided")

    system_prompt = build_estimator_prompt(shop)

    # Build multi-part content: text + 1–3 images
    content: List[dict] = [
        {
            "type": "text",
            "text": (
                "Here are the customer photos of vehicle damage. "
                "Give a clear, customer-friendly breakdown:\n"
                "1) Severity (Minor / Moderate / Severe)\n"
                "2) Damaged areas & damage types\n"
                "3) Simple repair steps\n"
                "4) Estimated CAD cost range\n\n"
                "Keep it under ~250 words."
            ),
        }
    ]

    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        })

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # vision-capable; server side handles it
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.4,
        max_tokens=700,
    )

    message = completion.choices[0].message
    return message.content or "I could not generate an estimate from these photos."


# ============================================================
# SIMPLE STATE (OPTIONAL FUTURE USE)
# ============================================================

# For now we don't track complex conversation state.
# Twilio sends a message → we reply once with plain text.


# ============================================================
# TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    """
    Twilio will POST here when an SMS/MMS comes in.

    URL in Twilio console:
    https://web-production-xxxxx.up.railway.app/sms-webhook?token=shop_miss_123
    """
    shop = get_shop_by_token(token)
    if not shop:
        # Return 200 so Twilio doesn't retry, but indicate misconfig.
        return PlainTextResponse(
            "Unknown shop token. Please contact the shop directly.",
            status_code=200,
        )

    form = await request.form()

    from_number = (form.get("From") or "").strip()
    body = (form.get("Body") or "").strip()
    num_media_str = form.get("NumMedia") or "0"

    try:
        num_media = int(num_media_str)
    except ValueError:
        num_media = 0

    # --------------------------------------------------------
    # 1) No images – send welcome & instructions
    # --------------------------------------------------------
    if num_media == 0:
        # First-touch welcome message
        text = (
            f"👋 Welcome to {shop.name}!\n\n"
            "Please send 1–3 clear photos of the vehicle damage "
            "(different angles and distances). "
            "Our AI estimator will review them and text you a preliminary repair estimate."
        )
        return PlainTextResponse(text)

    # --------------------------------------------------------
    # 2) We have at least one image – run AI estimator
    # --------------------------------------------------------
    image_urls: List[str] = []
    for i in range(num_media):
        key = f"MediaUrl{i}"
        url = form.get(key)
        if url:
            image_urls.append(url)

    if not image_urls:
        return PlainTextResponse(
            "⚠️ I received your message but couldn't read the photos. "
            "Please try sending them again.",
            status_code=200,
        )

    try:
        estimate_text = analyze_damage_with_vision(image_urls, shop)

        reply = (
            "📸 Thanks! We received your photos.\n\n"
            "📊 *AI Damage Estimate (preview)*\n\n"
            f"{estimate_text}\n\n"
            "ℹ️ This is a visual, preliminary estimate only. "
            "Final pricing may change after an in-person inspection at the shop."
        )

        return PlainTextResponse(reply, status_code=200)

    except Exception as e:
        # Log server-side for debugging
        print("AI Processing Error:", repr(e))

        error_reply = (
            "⚠️ AI Processing Error: I couldn't analyze your photos this time. "
            "Please try again in a few minutes, or contact the shop directly."
        )
        return PlainTextResponse(error_reply, status_code=200)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI estimator running"}
