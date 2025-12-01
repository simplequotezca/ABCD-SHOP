import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse

from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base


# ============================================================
# Environment + OpenAI client
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SHOPS_JSON = os.getenv("SHOPS_JSON")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
if not SHOPS_JSON:
    raise RuntimeError("SHOPS_JSON is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Database setup (minimal estimates table)
# ============================================================

Base = declarative_base()


class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, index=True)
    phone = Column(String, nullable=False)
    shop_id = Column(String, nullable=False)
    image_url = Column(Text, nullable=False)     # store JSON list of URLs
    ai_analysis = Column(Text, nullable=False)   # full text sent to user
    created_at = Column(DateTime, nullable=False, index=True)


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# create table if it does not exist
Base.metadata.create_all(bind=engine)


# ============================================================
# Shop config loader
# ============================================================

class Shop(Dict[str, Any]):
    id: str
    name: str
    webhook_token: str


def load_shops() -> Dict[str, Shop]:
    try:
        raw = json.loads(SHOPS_JSON)
        by_token: Dict[str, Shop] = {}
        for item in raw:
            token = item.get("webhook_token")
            if not token:
                continue
            by_token[token] = item
        return by_token
    except Exception as e:
        raise RuntimeError(f"Failed to parse SHOPS_JSON: {e}")


SHOPS = load_shops()


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


# ============================================================
# OpenAI helper
# ============================================================

def build_system_prompt(shop_name: str) -> str:
    return (
        "You are an AI collision damage estimator helping an auto body shop respond to customers by SMS.\n"
        f"Shop name: {shop_name}.\n\n"
        "INSTRUCTIONS:\n"
        "- Look closely at all uploaded photos of a vehicle collision (1–3 images).\n"
        "- Describe the visible damage in plain language from the driver's point of view.\n"
        "- Classify overall severity as one of: Minor, Moderate, Severe.\n"
        "- Give an estimated price *range* in Canadian dollars for Ontario 2025 body shop pricing.\n"
        "  Use realistic ranges for professional collision repair (parts + paint + labour).\n"
        "- Keep the reply under 900 characters so it fits nicely in 1–2 SMS messages.\n"
        "- Be clear that this is a preliminary visual estimate and not a final repair bill.\n\n"
        "RESPONSE FORMAT (very important):\n"
        "1) One short line: Severity and price range.\n"
        "2) 2–4 bullet points describing exactly which panels and parts seem damaged.\n"
        "3) One short line reminding them this is a preliminary estimate only.\n"
    )


def merge_assistant_content(message) -> str:
    """
    OpenAI chat responses can return content as a string or a list of parts.
    This helper always gives us a plain string.
    """
    content = message.content
    if isinstance(content, str):
        return content
    # content may be a list of {"type": "text", "text": "..."} etc.
    parts: List[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(p for p in parts if p)


def analyze_damage_with_openai(image_urls: List[str], shop_name: str) -> str:
    """
    Call GPT-4o-mini with image URLs and get a nice SMS-sized analysis string.
    We use the chat.completions endpoint which supports vision via image_url.
    """
    system_prompt = build_system_prompt(shop_name)

    # First message: system instructions + user content
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Customer has provided the following collision photos. "
                        "Carefully inspect each image and follow the response format."
                    ),
                },
            ],
        },
    ]

    # Attach each image correctly as image_url entries (no 'input_image' anywhere)
    for url in image_urls:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=700,
    )

    full_text = merge_assistant_content(completion.choices[0].message)
    return full_text.strip()


# ============================================================
# Twilio webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()

    phone: str = form.get("From", "")
    body: str = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", 0) or 0)

    # Identify shop by token in query string
    token = request.query_params.get("token")
    if not token or token not in SHOPS:
        # Respond with plain text so Twilio doesn't retry forever
        return PlainTextResponse("Invalid shop token.", status_code=403)

    shop = SHOPS[token]
    shop_name = shop.get("name", "our collision centre")

    twilio_resp = MessagingResponse()

    # No images: send friendly welcome / instructions
    if num_media == 0:
        lower = body.lower()
        if not body or any(greet in lower for greet in ["hi", "hello", "hey"]):
            msg = (
                f"👋 Welcome to {shop_name}!\n\n"
                "Please send 1–3 clear photos of your vehicle damage (different angles if possible). "
                "I’ll analyze them and text you a free preliminary AI estimate."
            )
        else:
            msg = (
                "To get your free AI estimate, please reply with 1–3 clear photos of the damage "
                "(close-up + wider angle if you can)."
            )

        twilio_resp.message(msg)
        return Response(str(twilio_resp), media_type="application/xml")

    # Collect image URLs from Twilio
    image_urls: List[str] = []
    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        media_type = form.get(f"MediaContentType{i}", "")
        if media_url and media_type.startswith("image/"):
            image_urls.append(media_url)

    if not image_urls:
        twilio_resp.message(
            "⚠️ I couldn't detect any valid images in your message. "
            "Please try sending the photos again."
        )
        return Response(str(twilio_resp), media_type="application/xml")

    # Acknowledge receipt quickly
    twilio_resp.message(
        "📸 Thanks! We received your photos.\n\n"
        f"Our AI estimator for {shop_name} is analyzing the damage now — "
        "you’ll receive a detailed breakdown shortly."
    )

    # Generate analysis with OpenAI
    try:
        analysis_text = analyze_damage_with_openai(image_urls, shop_name)
    except Exception:
        twilio_resp.message(
            "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
            "Please try again in a few minutes."
        )
        return Response(str(twilio_resp), media_type="application/xml")

    # Save to DB
    try:
        db = SessionLocal()
        estimate = Estimate(
            id=str(uuid.uuid4()),
            phone=phone,
            shop_id=shop.get("id", token),
            image_url=json.dumps(image_urls),
            ai_analysis=analysis_text,
            created_at=datetime.utcnow(),
        )
        db.add(estimate)
        db.commit()
    except Exception:
        # If DB fails we still send the user their estimate
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass

    # Send AI analysis as a follow-up message
    twilio_resp.message(analysis_text)
    return Response(str(twilio_resp), media_type="application/xml")
