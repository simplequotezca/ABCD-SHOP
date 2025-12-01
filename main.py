import os
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

# ============================================================
# FastAPI + OpenAI Setup
# ============================================================

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# AI Damage Analysis
# ============================================================

async def analyze_damage(image_urls: list):
    """
    Analyze uploaded vehicle damage using gpt-4o-mini (VISION ENABLED)
    """

    system_prompt = (
        "You are an elite auto-body estimator. "
        "Using the provided images, output STRICT JSON ONLY:\n\n"
        "{\n"
        '  "severity": "minor | moderate | severe",\n'
        '  "estimated_cost_min": number,\n'
        '  "estimated_cost_max": number,\n'
        '  "damaged_areas": [list],\n'
        '  "damage_types": [list],\n'
        '  "summary": "2–4 sentence summary"\n'
        "}\n\n"
        "DO NOT add comments or text outside JSON."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze these vehicle images."},
                *[
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # ✅ THIS ONE 100% WORKS
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )

        raw = response.choices[0].message["content"].strip()

        # Remove code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()

        return json.loads(raw)

    except Exception as e:
        print("AI ERROR:", e)
        return None


# ============================================================
# SMS Webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()

    num_media = int(form.get("NumMedia", "0"))

    reply = MessagingResponse()

    # ---------------------------------------
    # FIRST MESSAGE — NO IMAGES
    # ---------------------------------------
    if num_media == 0:
        reply.message(
            "📸 Welcome to Mississauga Collision Centre!\n\n"
            "Please send 1–3 photos of the vehicle damage for your instant AI estimate."
        )
        return Response(str(reply), media_type="application/xml")

    # ---------------------------------------
    # IMAGES RECEIVED
    # ---------------------------------------
    image_urls = []
    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        if url:
            image_urls.append(url)

    reply.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is reviewing the damage now — you'll get the breakdown shortly."
    )

    # RUN AI
    result = await analyze_damage(image_urls)

    if result is None:
        reply.message(
            "⚠️ AI Processing Error: Couldn't analyze your photos. Please try again in a few minutes."
        )
        return Response(str(reply), media_type="application/xml")

    # ---------------------------------------
    # FORMAT AI OUTPUT
    # ---------------------------------------
    out = (
        "🛠 **AI Damage Estimate**\n\n"
        f"**Severity:** {result['severity'].title()}\n"
        f"**Estimated Cost:** ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n\n"
        f"**Areas Affected:** {', '.join(result['damaged_areas'])}\n"
        f"**Damage Types:** {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n\n"
        "Reply *1* to confirm estimate or *2* to upload more photos."
    )

    reply.message(out)
    return Response(str(reply), media_type="application/xml")
