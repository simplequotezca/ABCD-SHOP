import os
import json
import uuid
import base64
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from sqlalchemy import (
    create_engine, Column, String, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================================================
# ENVIRONMENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

if not SHOPS_JSON:
    raise RuntimeError("Missing SHOPS_JSON")

if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)


# Load shop config
SHOPS = {shop["webhook_token"]: shop for shop in json.loads(SHOPS_JSON)}

# ============================================================
# DATABASE SETUP — SIMPLE LOGGING ONLY
# ============================================================

Base = declarative_base()

class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True)
    phone = Column(String)              # <-- matches your DB column
    shop_id = Column(String)
    image_url = Column(Text)            # <-- matches your DB column
    ai_analysis = Column(Text)          # <-- matches your DB column
    created_at = Column(DateTime)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

# ============================================================
# AI ANALYSIS — MULTI-IMAGE + ULTRA DETAILED DAMAGE BREAKDOWN
# ============================================================

def analyze_damage(image_urls: List[str]) -> Dict[str, Any]:
    """
    Sends images to GPT-4o Vision with the A2 Ultra-Detailed prompt.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a master auto-collision estimator with 20+ years of experience. "
                       "You ALWAYS use the driver's point of view for left/right. "
                       "You provide extremely detailed damage analysis and realistic Ontario 2025 pricing."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Perform a complete A2 ULTRA-DETAILED COLLISION ESTIMATE.\n"
                        "Include:\n"
                        "- Exact damaged panels\n"
                        "- Cracks, dents, deformation severity\n"
                        "- Hidden damage probability\n"
                        "- Structural concerns\n"
                        "- Repair vs replace decisions\n"
                        "- Realistic Ontario 2025 cost range\n"
                        "- Labour hours estimate\n"
                        "- Materials required\n"
                        "- Safety concerns\n"
                        "- Final customer-friendly summary\n"
                        "\nProcess ALL images together."
                    )
                }
            ]
        }
    ]

    # attach each image correctly — NO nested objects
    for url in image_urls:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=700
        )

        full_text = response.choices[0].message["content"]

        return {
            "full_text": full_text,
            "severity": "Moderate",  # still useful for UI
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    phone = form.get("From")
    body = form.get("Body", "").strip()
    num_media = int(form.get("NumMedia", 0))

    token = request.query_params.get("token")
    if token not in SHOPS:
        return PlainTextResponse("Invalid shop token.", status_code=403)

    shop = SHOPS[token]

    resp = MessagingResponse()

    # 1) FIRST MESSAGE — Friendly welcome + ask for photos
    if num_media == 0:
        resp.message(
            f"👋 Welcome to {shop['name']}!\n\n"
            "Please send 1–3 photos of your vehicle damage and I’ll generate a free AI estimate."
        )
        return PlainTextResponse(str(resp))

    # 2) PHOTO RECEIVED MESSAGE
    resp.message(
        "📸 Thanks! We received your photos.\n\n"
        f"Our AI estimator for {shop['name']} is analyzing the damage now — "
        "you’ll receive a detailed breakdown shortly."
    )

    # Extract image URLs
    image_urls = []
    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        if media_url:
            image_urls.append(media_url)

    # Run AI analysis
    analysis = analyze_damage(image_urls)

    # If AI failed
    if "error" in analysis:
        resp.message(
            "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
            "Please try again in a few minutes."
        )
        return PlainTextResponse(str(resp))

    # Save to DB
    session = SessionLocal()
    try:
        entry = Estimate(
            id=str(uuid.uuid4()),
            phone=phone,
            shop_id=token,
            image_url=json.dumps(image_urls),
            ai_analysis=analysis["full_text"],
            created_at=datetime.utcnow()
        )
        session.add(entry)
        session.commit()
    except Exception as e:
        session.rollback()
        resp.message(f"⚠️ Database Error: {e}")
        return PlainTextResponse(str(resp))
    finally:
        session.close()

    # 3) SEND CUSTOMER THE ULTRA-DETAILED ANALYSIS
    resp.message(analysis["full_text"])

    # 4) BOOKING MESSAGE
    resp.message(
        "📅 If you'd like to book an appointment, reply with:\n\n"
        "**BOOK** followed by your name, email, phone number, and preferred date/time.\n\n"
        "Example:\n"
        "BOOK John Doe john@example.com 416-555-0000 Tomorrow at 3pm"
    )

    return PlainTextResponse(str(resp))
