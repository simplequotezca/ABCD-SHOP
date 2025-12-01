import os
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# ============================================================
# FastAPI + OpenAI client
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Helper function — call AI estimator
# ============================================================

async def analyze_damage(image_urls: list):
    """
    Sends the images to GPT-4o-mini-vision and returns parsed JSON or None.
    """

    prompt = (
        "You are an elite auto-body AI damage estimator. "
        "Analyze all provided vehicle images and produce STRICT JSON ONLY in the following format:\n\n"
        "{\n"
        '  "severity": "minor | moderate | severe",\n'
        '  "estimated_cost_min": number,\n'
        '  "estimated_cost_max": number,\n'
        '  "damaged_areas": [list of areas],\n'
        '  "damage_types": [list],\n'
        '  "summary": "2–4 sentence customer-friendly summary"\n'
        "}\n\n"
        "Do not add commentary or text before or after the JSON."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-vision",
            messages=messages,
            max_tokens=500,
            temperature=0.2,
        )

        raw = response.choices[0].message["content"].strip()

        # Strict JSON extraction
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()

        return json.loads(raw)

    except Exception as e:
        print("AI ERROR:", str(e))
        return None


# ============================================================
# SMS Webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "").strip().lower()
    num_media = int(form.get("NumMedia", "0"))

    reply = MessagingResponse()

    # ------------------------------------------------------------
    # FIRST MESSAGE (no images)
    # ------------------------------------------------------------
    if num_media == 0:
        msg = reply.message(
            "📸 Welcome to Mississauga Collision Centre!\n\n"
            "Please send *1–3 photos* of the vehicle damage for your instant AI estimate."
        )
        return Response(str(reply), media_type="application/xml")

    # ------------------------------------------------------------
    # IMAGE RECEIVED → Process
    # ------------------------------------------------------------
    image_urls = []
    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        if media_url:
            image_urls.append(media_url)

    # Acknowledge first
    reply.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is reviewing the damage now — you'll get the breakdown shortly."
    )

    # ------------------------------------------------------------
    # Call AI
    # ------------------------------------------------------------
    ai_result = await analyze_damage(image_urls)

    if ai_result is None:
        reply.message(
            "⚠️ AI Processing Error: We couldn't analyze your photos this time.\n"
            "Please try again in a few minutes."
        )
        return Response(str(reply), media_type="application/xml")

    # ------------------------------------------------------------
    # Build final estimate message
    # ------------------------------------------------------------
    out = (
        "🛠 **AI Damage Estimate**\n\n"
        f"**Severity:** {ai_result.get('severity').title()}\n"
        f"**Estimated Cost:** ${ai_result.get('estimated_cost_min')} – ${ai_result.get('estimated_cost_max')}\n\n"
        f"**Areas Affected:** {', '.join(ai_result.get('damaged_areas', []))}\n"
        f"**Damage Types:** {', '.join(ai_result.get('damage_types', []))}\n\n"
        f"{ai_result.get('summary')}\n\n"
        "Reply *1* to confirm this estimate is accurate, or *2* to send more photos."
    )

    reply.message(out)

    return Response(str(reply), media_type="application/xml")
