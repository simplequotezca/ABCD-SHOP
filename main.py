import os
import json
import httpx
import base64
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Download Twilio Image (supports redirects)
# ============================================================

async def download_twilio_image(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as req:
        r = await req.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        r.raise_for_status()
        return r.content

# ============================================================
# Analyze Damage
# ============================================================

async def analyze_damage(image_bytes_list):
    system_prompt = (
        "You are an auto-body estimator. Return STRICT JSON ONLY:\n\n"
        "{\n"
        '  "severity": "minor | moderate | severe",\n'
        '  "estimated_cost_min": number,\n'
        '  "estimated_cost_max": number,\n'
        '  "damaged_areas": [list],\n'
        '  "damage_types": [list],\n'
        '  "summary": "2–4 sentences"\n'
        "}"
    )

    content_list = [
        {"type": "text", "text": "Analyze the vehicle damage in the attached images."}
    ]

    for img_bytes in image_bytes_list:
        b64 = base64.b64encode(img_bytes).decode()
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ],
            max_tokens=600,
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()

        # Remove markdown wrappers if present
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

    if num_media == 0:
        reply.message(
            "📸 Welcome to Mississauga Collision Centre!\n\n"
            "Please send 1–3 clear photos of the vehicle damage for your instant AI estimate."
        )
        return Response(str(reply), media_type="application/xml")

    # Download images
    images = []
    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        try:
            img = await download_twilio_image(url)
            images.append(img)
        except Exception as e:
            print("Image download error:", e)

    reply.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is reviewing the damage now — you'll get the breakdown shortly."
    )

    # Run AI
    result = await analyze_damage(images)

    if result is None:
        reply.message(
            "⚠️ AI Processing Error: Couldn't analyze your photos. Please try again shortly."
        )
        return Response(str(reply), media_type="application/xml")

    out = (
        "🛠 **AI Damage Estimate**\n\n"
        f"**Severity:** {result['severity']}\n"
        f"**Estimated Cost:** ${result['estimated_cost_min']} – ${result['estimated_cost_max']}\n\n"
        f"**Areas:** {', '.join(result['damaged_areas'])}\n"
        f"**Damage Types:** {', '.join(result['damage_types'])}\n\n"
        f"{result['summary']}\n\n"
        "Reply *1* to confirm or *2* to upload more photos."
    )

    reply.message(out)
    return Response(str(reply), media_type="application/xml")
