import os
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------
# Utility: Download image bytes from Twilio MMS URL
# ---------------------------------------------------
async def download_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=20.0, verify=False) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


# ---------------------------------------------------
# AI Damage Analyzer (gpt-4.1-vision)
# ---------------------------------------------------
async def analyze_damage(image_bytes_list: list):

    prompt = (
        "You are an expert auto-body estimator. Analyze the vehicle damage from the photos "
        "and return *ONLY* a JSON object with the following keys:\n\n"
        "{\n"
        '  "severity": "minor | moderate | severe",\n'
        '  "cost_min": 0,\n'
        '  "cost_max": 0,\n'
        '  "panels": ["list of damaged areas"],\n'
        '  "damage_types": ["dent, crack, scrape, deformation, misalignment, etc"],\n'
        '  "summary": "2–4 sentence explanation"\n'
        "}\n\n"
        "Do not include anything outside JSON."
    )

    # Build the model input
    input_blocks = [{"type": "input_text", "text": prompt}]

    for img_bytes in image_bytes_list:
        input_blocks.append({
            "type": "input_image",
            "image": img_bytes
        })

    # Call model
    result = client.responses.create(
        model="gpt-4.1-vision",
        input=input_blocks,
        max_output_tokens=500
    )

    raw = result.output_text
    return json.loads(raw)


# ---------------------------------------------------
# Twilio Webhook
# ---------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    token = request.query_params.get("token", "")

    # Security check
    if token != "shop_miss_123":
        return PlainTextResponse("Invalid token", status_code=403)

    num_media = int(form.get("NumMedia", "0"))
    from_number = form.get("From", "")

    resp = MessagingResponse()

    # First message from customer (no media)
    if num_media == 0:
        resp.message(
            "📸 Welcome to Mississauga Collision Centre!\n\n"
            "Please send 1–3 photos of the vehicle damage for your instant AI estimate."
        )
        return PlainTextResponse(str(resp))

    # Acknowledge we received photos
    resp.message(
        "📸 Thanks! We received your photos.\n\n"
        "Our AI estimator is analyzing the damage now — you'll receive a "
        "detailed breakdown shortly."
    )

    # Process images
    try:
        images_bytes = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            images_bytes.append(await download_image(url))

        analysis = await analyze_damage(images_bytes)

        # Format final message
        result_msg = (
            "🛠 **AI Damage Estimate**\n\n"
            f"Severity: **{analysis['severity'].title()}**\n"
            f"Estimated Cost: **${analysis['cost_min']}–${analysis['cost_max']}**\n"
            f"Damaged Areas: {', '.join(analysis['panels'])}\n"
            f"Damage Types: {', '.join(analysis['damage_types'])}\n\n"
            f"{analysis['summary']}"
        )

        resp.message(result_msg)
        return PlainTextResponse(str(resp))

    except Exception as e:
        print("AI ERROR:", str(e))
        resp.message(
            "⚠️ AI Processing Error: We couldn't analyze your photos this time. "
            "Please try again in a few minutes."
        )
        return PlainTextResponse(str(resp))
