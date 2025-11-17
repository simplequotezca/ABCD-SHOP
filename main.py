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

# ----------------- Download + Base64 Encoding -----------------
def download_and_encode(image_url: str) -> str:
    img_bytes = requests.get(image_url).content
    return base64.b64encode(img_bytes).decode("utf-8")

# ----------------- AI Damage Analyzer -------------------------
def analyze_damage(image_b64: str):
    prompt = """
You are an auto body estimator AI. Analyze vehicle damage in detail.

Return:
- Severity (Minor/Moderate/Severe/Total Loss)
- Damaged Panels
- Damage Types
- Estimated Cost (CAD)

Be specific. No '-' responses. Always give results.
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
        max_output_tokens=400
    )

    return response.output_text

# ----------------- Twilio Webhook ------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "")
    image_url = form.get("MediaUrl0")

    reply = MessagingResponse()

    if image_url:
        try:
            encoded = download_and_encode(image_url)
            analysis = analyze_damage(encoded)
            estimate_id = str(uuid.uuid4())[:12]

            reply.message(
                f"AI Damage Estimate for {SHOP_NAME}\n\n"
                f"{analysis}\n\n"
                f"Estimate ID: {estimate_id}\n\n"
                "Reply with 1, 2, or 3 to book:\n"
                "1) Tue 9:00 AM\n"
                "2) Tue 11:00 AM\n"
                "3) Tue 2:00 PM"
            )
        except Exception as e:
            reply.message(f"Error: {e}")

        return Response(content=str(reply), media_type="application/xml")

    reply.message("Please send a photo of the vehicle damage.")
    return Response(content=str(reply), media_type="application/xml")

# ----------------- RUN SERVER for Railway ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
