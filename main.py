import os
import uuid
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SHOP_NAME = os.getenv("SHOP_NAME", "SJ Auto Body")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# -------------- Helper: Download image & convert to Base64 ----------------
def download_and_encode(image_url: str) -> str:
    img_bytes = requests.get(image_url).content
    return base64.b64encode(img_bytes).decode("utf-8")

# -------------- AI Damage Analysis (REAL GPT-4O Vision) -------------------
def analyze_damage(image_b64: str):
    prompt = """
You are an auto body damage estimator AI.

Analyze the vehicle damage in the image and return the following fields IN CLEAR, SIMPLE TEXT:

- Severity (Minor, Moderate, Severe, Total Loss)
- Detected Panels (list actual car panels)
- Damage Types (scratches, dents, cracks, bumper misalignment, crease, deep gouge, etc.)
- Estimated Repair Cost (Ontario 2025 prices)

Be very specific. Do NOT output dashes. Always give real findings even if approximate.
"""

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this vehicle damage:"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            }
        ],
        max_output_tokens=450,
    )

    return response.output_text

# -------------- SMS Webhook -------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "")
    image_url = form.get("MediaUrl0")
    from_number = form.get("From")

    reply = MessagingResponse()

    if image_url:
        try:
            encoded_img = download_and_encode(image_url)

            analysis = analyze_damage(encoded_img)

            estimate_id = str(uuid.uuid4())[:12]

            final_msg = f"""
AI Damage Estimate for {SHOP_NAME}

{analysis}

Estimate ID:
{estimate_id}

Reply with a number to book an appointment:
1) Tue Nov 18 at 09:00 AM
2) Tue Nov 18 at 11:00 AM
3) Tue Nov 18 at 02:00 PM
"""

            reply.message(final_msg)

        except Exception as e:
            reply.message(f"Error analyzing image: {e}")

        return Response(content=str(reply), media_type="application/xml")

    # If no image sent
    reply.message("Please send a clear photo of the vehicle damage.")
    return Response(content=str(reply), media_type="application/xml")
