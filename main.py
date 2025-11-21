import os
import json
import time
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

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
    """
    SHOPS_JSON example (Railway env var):

    [
      {"id": "mississauga", "name": "Mississauga Collision Centre", "webhook_token": "shop_miss_123"},
      {"id": "brampton", "name": "Brampton Collision Centre", "webhook_token": "shop_bram_456"}
    ]
    """
    raw = os.getenv("SHOPS_JSON", "[]")
    try:
        data = json.loads(raw)
        shops: Dict[str, Shop] = {}
        for item in data:
            shop = Shop(**item)
            shops[shop.webhook_token] = shop
        return shops
    except Exception as e:
        print("Failed to load SHOPS_JSON:", e)
        return {}


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()

# ============================================================
# Simple in-memory session store (good enough for MVP)
# ============================================================

SessionData = Dict[str, Any]
SESSIONS: Dict[str, SessionData] = {}
SESSION_TTL_SECONDS = 20 * 60  # 20 minutes


def session_key(shop: Shop, from_number: str) -> str:
    return f"{shop.id}:{from_number}"


def cleanup_sessions() -> None:
    now = time.time()
    expired_keys = [
        key for key, data in SESSIONS.items()
        if now - data.get("timestamp", 0) > SESSION_TTL_SECONDS
    ]
    for key in expired_keys:
        del SESSIONS[key]


def get_session(shop: Shop, from_number: str) -> Optional[SessionData]:
    cleanup_sessions()
    key = session_key(shop, from_number)
    data = SESSIONS.get(key)
    if not data:
        return None
    if time.time() - data.get("timestamp", 0) > SESSION_TTL_SECONDS:
        del SESSIONS[key]
        return None
    return data


def save_session(shop: Shop, from_number: str, data: SessionData) -> None:
    data["timestamp"] = time.time()
    SESSIONS[session_key(shop, from_number)] = data


def clear_session(shop: Shop, from_number: str) -> None:
    key = session_key(shop, from_number)
    if key in SESSIONS:
        del SESSIONS[key]


# ============================================================
# AI helpers
# ============================================================

def call_prescan_model(image_urls: List[str]) -> str:
    """
    Quick pre-scan summary of visible damage only.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an auto body pre-scan estimator. "
                "You only comment on visible exterior vehicle damage. "
                "Keep it short and plain-language for Ontario customers."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "These are photos of vehicle damage. "
                        "Describe the visible damage in 2–3 sentences. "
                        "Do NOT give prices here. Focus only on where and how bad the damage looks."
                    ),
                },
                *[
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            ],
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def call_full_estimate_model(
    image_urls: List[str],
    prescan_summary: str,
    shop: Shop
) -> str:
    """
    Detailed Ontario 2025-style estimate (Option A).
    """
    system_prompt = (
        "You are a professional auto body estimator in Ontario, Canada (2025). "
        "You estimate collision repair costs for a body shop. "
        "You ONLY use Canadian/Ontario style labour rates and terminology.\n\n"
        "Output a clear, friendly estimate for a customer, using this structure:\n\n"
        "1) Short damage summary (2–3 sentences)\n"
        "2) Severity: Minor / Moderate / Major\n"
        "3) Estimated cost range in CAD (e.g., $1,800–$2,300)\n"
        "4) Breakdown:\n"
        "   - Parts (list major parts with approximate amounts)\n"
        "   - Labour (body hours, paint hours, R&I hours with typical Ontario 2025 rates: "
        "body ~$75–$95/hr, paint ~$80–$100/hr)\n"
        "   - Paint & materials (flat amount or per-hour)\n"
        "   - Other: shop supplies / environmental fees\n"
        "5) Notes: mention if there could be hidden damage or frame/structural concerns.\n"
        "6) Insurance suggestion: briefly say if it's typically an insurance claim "
        "or sometimes paid out-of-pocket, depending on severity.\n\n"
        "Important rules:\n"
        "- Give a realistic but approximate RANGE, not a single exact number.\n"
        "- Assume OEM or high-quality aftermarket parts, NOT junkyard.\n"
        "- Do NOT promise final prices. Always say it's a visual estimate and "
        f"final numbers require in-person inspection at {shop.name}.\n"
    )

    user_text = (
        "Here are vehicle damage photos and a pre-scan summary from a previous step.\n\n"
        f"Pre-scan summary:\n{prescan_summary}\n\n"
        "Based on the images and the summary, generate the detailed estimate following the structure."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                *[
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            ],
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# Core SMS webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Main Twilio webhook endpoint.

    Twilio should call:
    https://<your-domain>/sms-webhook?token=SHOP_WEBHOOK_TOKEN
    """
    token = request.query_params.get("token")
    shop: Optional[Shop] = SHOPS_BY_TOKEN.get(token) if token else None

    reply = MessagingResponse()

    if not shop:
        reply.message(
            "This number is not configured correctly. "
            "Please contact the body shop directly."
        )
        return Response(content=str(reply), media_type="application/xml")

    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From", "")
        num_media = int(form.get("NumMedia", "0") or "0")

        cleanup_sessions()
        lower_body = body.lower()

        # --------------------------------------------------------
        # CASE 1 — MMS received → PRE-SCAN
        # --------------------------------------------------------
        if num_media > 0:
            image_urls: List[str] = []
            for i in range(num_media):
                media_url = form.get(f"MediaUrl{i}")
                if media_url:
                    image_urls.append(media_url)

            if not image_urls:
                reply.message(
                    "I couldn't read the photo URL from your message. "
                    "Please try sending the photos again."
                )
                return Response(content=str(reply), media_type="application/xml")

            # Call AI for quick pre-scan
            try:
                prescan_summary = call_prescan_model(image_urls)
            except Exception:
                reply.message(
                    "Sorry — I had trouble processing the photos. "
                    "Please try again with 1–3 clear photos of the damage."
                )
                return Response(content=str(reply), media_type="application/xml")

            # Save session state
            save_session(
                shop,
                from_number,
                {
                    "step": "awaiting_prescan_confirmation",
                    "images": image_urls,
                    "prescan_summary": prescan_summary,
                },
            )

            reply.message(
                f"Quick Pre-Scan from {shop.name}:\n\n"
                f"{prescan_summary}\n\n"
                "If this looks right, reply 1.\n"
                "If it's off, reply 2 and you can send new photos."
            )
            return Response(content=str(reply), media_type="application/xml")

        # --------------------------------------------------------
        # CASE 2 — No media: handle conversation steps
        # --------------------------------------------------------
        session = get_session(shop, from_number)
        step = session["step"] if session else "idle"

        # --- Step: user confirms pre-scan is correct ---
        if step == "awaiting_prescan_confirmation" and lower_body in {"1", "yes", "y"}:
            image_urls = session.get("images", [])
            prescan_summary = session.get("prescan_summary", "")

            if not image_urls:
                clear_session(shop, from_number)
                reply.message(
                    "I lost track of your photos. Please send 1–3 photos of the damage again."
                )
                return Response(content=str(reply), media_type="application/xml")

            try:
                estimate_text = call_full_estimate_model(
                    image_urls=image_urls,
                    prescan_summary=prescan_summary,
                    shop=shop,
                )
            except Exception:
                reply.message(
                    "Sorry — something went wrong while generating the estimate. "
                    "Please try sending the photos again, or contact the shop directly."
                )
                clear_session(shop, from_number)
                return Response(content=str(reply), media_type="application/xml")

            clear_session(shop, from_number)

            reply.message(
                f"Here’s your detailed visual estimate (Ontario 2025 pricing):\n\n"
                f"{estimate_text}\n\n"
                "This is a visual estimate only. Final pricing may change after an in-person "
                f"inspection at {shop.name}."
            )
            return Response(content=str(reply), media_type="application/xml")

        # --- Step: user says pre-scan is off ---
        if step == "awaiting_prescan_confirmation" and lower_body in {"2", "no", "n"}:
            clear_session(shop, from_number)
            reply.message(
                "No problem — sometimes photos can be tricky.\n\n"
                "Please send 1–3 clear photos of the damage from different angles, "
                "and I’ll redo the pre-scan."
            )
            return Response(content=str(reply), media_type="application/xml")

        # --------------------------------------------------------
        # CASE 3 — New conversation / generic text
        # --------------------------------------------------------
        if lower_body in {"hi", "hello", "hey", "estimate", "quote", ""}:
            reply.message(
                f"Hi from {shop.name}!\n\n"
                "To get an AI-powered damage estimate:\n"
                "1) Send 1–3 clear photos of the damaged area.\n"
                "2) I’ll send a quick Pre-Scan.\n"
                "3) Reply 1 if it looks right, or 2 if it’s off.\n"
                "4) Then I’ll send your full Ontario 2025 cost estimate.\n\n"
                "You can start by sending photos now."
            )
            return Response(content=str(reply), media_type="application/xml")

        # Fallback for other messages
        reply.message(
            "To get an AI-powered damage estimate, please send 1–3 clear photos "
            "of the damaged area, and I’ll take it from there."
        )
        return Response(content=str(reply), media_type="application/xml")

    except Exception as e:
        # Log server-side
        print("Unexpected error in /sms-webhook:", e)
        reply.message(
            "Sorry — something went wrong on my side. "
            "Please try again in a few minutes, or contact the shop directly."
        )
        return Response(content=str(reply), media_type="application/xml")


# ============================================================
# Simple health check
# ============================================================

@app.get("/health")
async def health():
    return PlainTextResponse("OK")


# Optional: for local testing with `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
