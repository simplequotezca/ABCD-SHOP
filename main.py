# ============================
# main.py — FINAL STABLE BUILD
# ============================

from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import requests
import base64
import os

app = FastAPI()

# ----------------------------
# ENVIRONMENT
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# WELCOME MESSAGE
# ----------------------------
WELCOME_MESSAGE = (
    "📸 Welcome to Mississauga Collision Centre!\n\n"
    "Please send 1–3 photos of the vehicle damage for your instant AI estimate."
)

# ----------------------------
# MAIN WEBHOOK
# ----------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "").strip()
    num_media = int(form.get("NumMedia", "0"))

    resp = MessagingResponse()

    # --------------------------------------------
    # FIRST MESSAGE → SEND WELCOME
    # --------------------------------------------
    if num_media == 0:
        resp.message(WELCOME_MESSAGE)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # --------------------------------------------
    # RECEIVE IMAGES
    # --------------------------------------------
    image_blocks = []

    try:
        for i in range(num_media):
            media_url = form.get(f"MediaUrl{i}")
            content_type = form.get(f"MediaContentType{i}", "image/jpeg")

            img = requests.get(media_url).content

            # NEW REQUIRED FORMAT — BASE64 ENCODED
            encoded = base64.b64encode(img).decode("utf-8")

            image_blocks.append({
                "type": "input_image",
                "image": {
                    "data": encoded,
                    "mime_type": content_type
                }
            })

    except Exception as e:
        resp.message(
            "⚠️ Error downloading your images. Please try again in a moment."
        )
        return PlainTextResponse(str(resp), media_type="application/xml")

    # --------------------------------------------
    # SEND ACKNOWLEDGEMENT IMMEDIATELY
    # --------------------------------------------
    ack = resp.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is analyzing the damage now — "
        "you’ll receive a detailed breakdown shortly."
    )

    # Continue processing AFTER responding to Twilio
    xml_reply = str(resp)

    # --------------------------------------------
    # PROCESS WITH OPENAI
    # --------------------------------------------
    try:
        system_prompt = (
            "You are a professional auto-body estimator. "
            "Analyze vehicle damage extremely accurately. "
            "Output ONLY JSON:\n\n"
            "{\n"
            '  "severity": "minor/moderate/severe",\n'
            '  "estimated_cost": [min, max],\n'
            '  "damaged_areas": ["..."],\n'
            '  "damage_types": ["..."],\n'
            '  "summary": "2–4 sentence explanation."\n'
            "}\n\n"
            "Do NOT include anything outside JSON."
        )

        user_text = (
            "Analyze the vehicle damage from these photos and produce the JSON described. "
            "Use typical Ontario collision repair pricing for 2025."
        )

        blocks = [
            {"type": "input_text", "text": user_text},
            *image_blocks
        ]

        ai = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": blocks
                }
            ]
        )

        result = ai.output_text.strip()

    except Exception as e:
        # If AI fails
        error_resp = MessagingResponse()
        error_resp.message(
            "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
            "Please try again in a few minutes."
        )
        return PlainTextResponse(str(error_resp), media_type="application/xml")

    # --------------------------------------------
    # SEND FINAL AI RESULT TO USER
    # --------------------------------------------
    final_resp = MessagingResponse()
    final_resp.message(f"🔧 AI Damage Estimate:\n\n{result}")

    return PlainTextResponse(xml_reply + str(final_resp), media_type="application/xml")
