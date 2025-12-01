import os
import uuid
import json
import base64
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse

import httpx
from openai import OpenAI

from sqlalchemy import (
    create_engine, Column, String, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================================================
# Environment + API Clients
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# ============================================================
# Database Model — Minimal Table
# ============================================================

class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, index=True)
    shop_id = Column(String, index=True)
    phone = Column(String)
    image_url = Column(Text)  # store JSON list of URLs
    ai_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()

# Load shop config from SHOPS_JSON env
SHOPS_JSON = os.getenv("SHOPS_JSON")
try:
    SHOPS = json.loads(SHOPS_JSON)
except:
    SHOPS = []


def find_shop_by_token(token: str):
    for shop in SHOPS:
        if shop["webhook_token"] == token:
            return shop
    return None


# ============================================================
# OPENAI DAMAGE ANALYSIS FUNCTION
# ============================================================

async def analyze_images(image_urls: list, shop_name: str):
    img_inputs = []

    # Download & convert images → base64
    async with httpx.AsyncClient() as client:
        for url in image_urls:
            # Follow Twilio redirect manually
            r = await client.get(url, follow_redirects=True)
            b64 = base64.b64encode(r.content).decode()
            img_inputs.append({
                "type": "input_image",
                "image_base64": b64
            })

    # Ultra-detailed structured reasoning
    prompt = f"""
You are an elite automotive repair estimator.

Generate:
- Severity (Minor/Moderate/Severe)
- Precise panels damaged
- Material & labour reasoning
- Cost range in CAD
- Clear final summary for the customer

Shop: {shop_name}
"""

    # OpenAI call
    ai = client.responses.create(
        model="gpt-4.1",
        reasoning={"effort": "medium"},
        input=[
            *img_inputs,
            {"role": "user", "content": prompt}
        ]
    )

    return ai.output_text


# ============================================================
# TWILIO SMS WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    token = request.query_params.get("token")

    shop = find_shop_by_token(token)
    if not shop:
        raise HTTPException(400, "Invalid shop token")

    phone = form.get("From", "")
    msg_body = form.get("Body", "")
    num_media = int(form.get("NumMedia", 0))

    resp = MessagingResponse()

    # FIRST MESSAGE (Greeting)
    if num_media == 0 and msg_body.strip().lower() in ["hi", "hello", "hey"]:
        m = resp.message(
            f"👋 Welcome to {shop['name']}!\n\n"
            f"Please send 1–3 photos of your vehicle damage for a free AI estimate."
        )
        return Response(content=str(resp), media_type="application/xml")

    # No images? Ask again
    if num_media == 0:
        m = resp.message(
            f"📸 Please send clear photos of the damage so I can analyze it."
        )
        return Response(content=str(resp), media_type="application/xml")

    # Extract image URLs with Twilio redirect
    img_urls = []
    for i in range(num_media):
        img_urls.append(form.get(f"MediaUrl{i}"))

    # Acknowledge reception
    resp.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is analyzing the damage now — you'll receive a detailed breakdown shortly."
    )

    # Process AI
    try:
        analysis_text = await analyze_images(img_urls, shop["name"])
    except Exception as e:
        resp.message(f"⚠️ AI Processing Error: {str(e)[:150]}")
        return Response(content=str(resp), media_type="application/xml")

    # Save to database
    db = SessionLocal()
    try:
        record = Estimate(
            id=str(uuid.uuid4()),
            shop_id=shop["id"],
            phone=phone,
            image_url=json.dumps(img_urls),
            ai_analysis=analysis_text
        )
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        resp.message(f"⚠️ Database Error: {str(e)}")
        return Response(content=str(resp), media_type="application/xml")
    finally:
        db.close()

    # Send final AI estimate
    resp.message(f"📝 AI Estimate:\n\n{analysis_text}")

    return Response(content=str(resp), media_type="application/xml")
