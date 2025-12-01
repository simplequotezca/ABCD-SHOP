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
# Environment + OpenAI Client
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# ============================================================
# Database Model (Minimal)
# ============================================================

class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, index=True)
    shop_id = Column(String, index=True)
    phone = Column(String)
    image_url = Column(Text)
    ai_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============================================================
# Load Multi-shop Config
# ============================================================

app = FastAPI()

try:
    SHOPS = json.loads(os.getenv("SHOPS_JSON", "[]"))
except:
    SHOPS = []

def find_shop_by_token(token):
    for shop in SHOPS:
        if shop["webhook_token"] == token:
            return shop
    return None

# ============================================================
# AI Damage Analysis
# ============================================================

async def analyze_images(image_urls, shop_name):
    img_inputs = []

    async with httpx.AsyncClient() as httpx_client:
        for url in image_urls:
            r = await httpx_client.get(url, follow_redirects=True)
            b64 = base64.b64encode(r.content).decode()

            img_inputs.append({
                "type": "input_image",
                "image_base64": b64
            })

    prompt = f"""
You are a certified auto damage estimator.

Provide:

- Severity (Minor / Moderate / Severe)
- Exact damaged panels
- Repair vs replace decisions
- CAD cost range
- A simple, customer-friendly summary
- Professional findings for the body shop

Shop: {shop_name}
"""

    completion = openai_client.responses.create(
        model="gpt-4.1",
        reasoning={"effort": "medium"},
        input=[
            *img_inputs,
            {"role": "user", "content": prompt}
        ]
    )

    return completion.output_text

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
    msg_body = form.get("Body", "").strip().lower()
    num_media = int(form.get("NumMedia", 0))

    resp = MessagingResponse()

    # Greeting
    if num_media == 0 and msg_body in ["hi", "hello", "hey", "start"]:
        resp.message(
            f"👋 Welcome to {shop['name']}!\n\n"
            "Please send 1–3 photos of your vehicle damage for a free AI estimate."
        )
        return Response(str(resp), media_type="application/xml")

    if num_media == 0:
        resp.message("📸 Please send clear photos of the damage.")
        return Response(str(resp), media_type="application/xml")

    # Extract image URLs
    img_urls = [form.get(f"MediaUrl{i}") for i in range(num_media)]

    # Confirm received
    resp.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is analyzing the damage now — you'll receive a detailed breakdown shortly."
    )

    # Process AI
    try:
        analysis_text = await analyze_images(img_urls, shop["name"])
    except Exception as e:
        resp.message(f"⚠️ AI Processing Error: {str(e)}")
        return Response(str(resp), media_type="application/xml")

    # Save record
    db = SessionLocal()
    try:
        row = Estimate(
            id=str(uuid.uuid4()),
            shop_id=shop["id"],
            phone=phone,
            image_url=json.dumps(img_urls),
            ai_analysis=analysis_text
        )
        db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        resp.message(f"⚠️ Database Error: {str(e)}")
        return Response(str(resp), media_type="application/xml")
    finally:
        db.close()

    # Final AI message
    resp.message(f"📝 AI Estimate:\n\n{analysis_text}")

    return Response(str(resp), media_type="application/xml")
