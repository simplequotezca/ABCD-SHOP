import os
import base64
import httpx
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

from openai import OpenAI
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base


# ---------------------------------------------------------
# ENV VARIABLES
# ---------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SHOPS_JSON = os.getenv("SHOPS_JSON")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")

client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()


# ---------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------

Base = declarative_base()


class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True)
    phone = Column(String)
    shop_id = Column(String)
    image_url = Column(Text)
    ai_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


# ---------------------------------------------------------
# SHOP CONFIG LOADING
# ---------------------------------------------------------

import json
shops: Dict[str, Any] = {}

try:
    data = json.loads(SHOPS_JSON)
    for s in data:
        shops[s["webhook_token"]] = s
except Exception:
    raise RuntimeError("Invalid SHOPS_JSON")


# ---------------------------------------------------------
# Download Twilio Media → Base64
# ---------------------------------------------------------

def download_and_encode_image(twilio_url: str) -> str:
    """
    Twilio media URLs require HTTP Basic Auth.
    We download the image → base64 encode it → pass to OpenAI.
    """
    with httpx.Client() as http:
        response = http.get(
            twilio_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


# ---------------------------------------------------------
# AI DAMAGE ANALYSIS WITH A2 ULTRA DETAILED PROMPT
# ---------------------------------------------------------

def analyze_damage(base64_img: str, shop: Dict[str, Any]) -> str:
    """
    Sends base64 image to OpenAI using correct Vision input format.
    """

    user_prompt = f"""
You are an AI Collision Estimator trained to analyze vehicle damage from images with professional,
industry-level accuracy. Produce an ultra-detailed, shop-ready evaluation based ONLY on visible evidence.

Follow this exact structure:

============================================================
🚗 VEHICLE DAMAGE SUMMARY
Describe overall impact direction, severity, collision scenario, and driver's POV orientation.

============================================================
📍 DAMAGED AREAS (EXTREMELY DETAILED)
List all visible damage with precise terminology:
- dents (depth, direction, severity)
- cracks, fractures, tears
- panel buckling, creasing, folding
- misalignment, shifted gaps, broken mounts
- structural deformation indicators

Specify for each:
- exact location (driver/passenger + upper/middle/lower)
- severity rating (minor/moderate/major/severe/structural)

============================================================
🛠️ STRUCTURAL & SAFETY RISKS
Analyze:
- frame rails, aprons, strut tower alignment
- radiator support deformation
- crash bar reinforcement impact
- wheel/suspension geometry, camber changes
- ADAS sensor damage risk (radar, cameras, parking sensors)
- possible airbag or pretensioner events
- driveability and safety rating (0–10)

============================================================
📦 PARTS LIKELY REQUIRED
List ALL components that likely need:
- replacement
- repair
- refinishing

Include:
bumper cover, reinforcement, fender, hood, grilles, lamps, brackets, shields, condensers, radiator support, mounts, hardware.

Mark each as:
**Replace**, **Repair**, or **Uncertain (requires teardown)**.

============================================================
🎨 PAINT & REFINISH OPERATIONS
Provide paint workflow:
- repair vs refinish
- prime/block/seal/base/clear steps
- blending into adjacent panels
- color match difficulty
- plastic/metal refinishing considerations

============================================================
⏱️ LABOR HOUR ESTIMATES (BODY / PAINT / MECH / FRAME)
Estimate realistic hours for:
- body repair per panel
- R&I / R&R operations
- mechanical work (cooling, suspension)
- frame pulls or measurements
- paint operations
- blend hours
- calibration (ADAS)

Format example:
- Fender (repair): 3.0 hr body, 1.5 hr paint
- Bumper (replace): 1.0 hr body, 1.0 hr paint
- Blend hood: 1.0 hr paint
- ADAS calibration: 1.5 hr mech

============================================================
⚠️ HIDDEN DAMAGE LIKELIHOOD
Provide probability estimations for:
- condenser/radiator puncture
- cooling fan shroud cracks
- reinforcement collapse
- apron damage
- alignment/suspension deviation
- sensor wiring damage
- headlamp mounting tab damage

Give each risk a % likelihood.

============================================================
📊 SEVERITY SCORE (0–10)
0 = purely cosmetic
10 = severe structural risk with frame involvement

============================================================
📄 CUSTOMER-FRIENDLY SUMMARY
Provide a simplified explanation customers can easily understand.

============================================================

Do NOT provide repair cost — that is handled later.
Just output the analysis in clean formatting.
"""

    payload = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "input_image": {"data": base64_img}
                    }
                ]
            }
        ],
        max_tokens=2300
    )

    return payload.choices[0].message["content"]


# ---------------------------------------------------------
# BACKGROUND WORKER: FULL ESTIMATE + SMS
# ---------------------------------------------------------

def process_estimate_async(estimate_id: str):
    db = SessionLocal()
    est = db.query(Estimate).filter(Estimate.id == estimate_id).first()
    if not est:
        db.close()
        return

    shop = shops[est.shop_id]

    try:
        base64_img = download_and_encode_image(est.image_url)
        ai_output = analyze_damage(base64_img, shop)

        est.ai_analysis = ai_output
        db.commit()

        # -------------------------------------------------
        # UPGRADED SECOND SMS (PROFESSIONAL SHOP-GRADE)
        # -------------------------------------------------
        twilio_client.messages.create(
            body=(
                f"🔍 **Your AI Damage Assessment is Ready**\n\n"
                f"{ai_output}\n\n"
                f"To receive a detailed repair cost range based on Ontario 2025 rates, reply:\n\n"
                f"➡️ 1 — Get Cost Estimate\n"
                f"➡️ 2 — Upload More Photos"
            ),
            from_=TWILIO_PHONE_NUMBER,
            to=est.phone
        )

    except Exception as e:
        twilio_client.messages.create(
            body=f"⚠️ AI Processing Error:\n{str(e)}",
            from_=TWILIO_PHONE_NUMBER,
            to=est.phone
        )

    db.close()


# ---------------------------------------------------------
# SMS WEBHOOK
# ---------------------------------------------------------

@app.post("/sms-webhook")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()

    from_number = form.get("From")
    body = (form.get("Body") or "").strip().lower()
    media_url = form.get("MediaUrl0")
    token = request.query_params.get("token")

    if token not in shops:
        return PlainTextResponse("Invalid shop token", status_code=403)

    shop = shops[token]

    # ---- New Image Arrived ----
    if media_url:
        est_id = str(uuid.uuid4())

        db = SessionLocal()
        record = Estimate(
            id=est_id,
            phone=from_number,
            shop_id=token,
            image_url=media_url
        )
        db.add(record)
        db.commit()
        db.close()

        # Initial SMS
        reply = MessagingResponse()
        reply.message(
            "📸 Thanks! We received your photos.\n\n"
            "Our AI estimator is analyzing the damage now — "
            "you’ll receive a detailed breakdown shortly."
        )

        background_tasks.add_task(process_estimate_async, est_id)
        return PlainTextResponse(str(reply), media_type="application/xml")


    # ---- User Requests Cost Estimate ----
    if body == "1":
        db = SessionLocal()
        est = (
            db.query(Estimate)
            .filter(Estimate.phone == from_number)
            .order_by(Estimate.created_at.desc())
            .first()
        )
        db.close()

        if not est or not est.ai_analysis:
            reply = MessagingResponse()
            reply.message("No recent estimate found.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        reply = MessagingResponse()
        reply.message(
            "💵 **Estimated Repair Cost Range (Ontario 2025 Rates)**\n\n"
            "• Minor: $350 – $650\n"
            "• Moderate: $900 – $1,600\n"
            "• Severe: $2,000 – $5,000+\n\n"
            "Reply BOOK to schedule an appointment."
        )
        return PlainTextResponse(str(reply), media_type="application/xml")


    # ---- Booking Flow ----
    if "book" in body:
        reply = MessagingResponse()
        reply.message(
            "🗓️ To schedule an appointment, please send:\n\n"
            "Full Name\n"
            "Phone Number\n"
            "Email\n"
            "Preferred Date & Time\n"
            "Vehicle Make/Model/Year"
        )
        return PlainTextResponse(str(reply), media_type="application/xml")


    # ---- Default Help Message ----
    reply = MessagingResponse()
    reply.message("Send 1–3 photos of the damage for an instant AI estimate.")
    return PlainTextResponse(str(reply), media_type="application/xml")
