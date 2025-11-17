
import uuid
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOP_NAME = os.getenv("SHOP_NAME", "Auto Body Shop")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------
# Helper: Download image + convert to Base64
# ---------------------------------------------------------
def download_and_encode(image_url: str) -> str:
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img_bytes = resp.content
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download/encode image: {e}")


# ---------------------------------------------------------
# AI Image Damage Analyzer (Ontario 2025 Calibration)
# ---------------------------------------------------------
def analyze_damage(image_b64: str) -> str:
    prompt = """
You are a certified Ontario (Canada) auto-body estimator (2025).
Analyze the uploaded vehicle damage and produce a detailed professional
estimate including:

- Damage Severity (Minor / Moderate / Severe / Total Loss)
- Specific damaged panels (never vague)
- Specific damage types (dent, crease, crack, scratch, deformation, etc.)
- Recommended repair methods (PDR, repaint, replacement, calibration, etc.)
- Estimated cost range in CAD (using 2025 Ontario market retail rates)
- Short explanation of why

Be specific and personalized to the photo. Never output placeholders.
    """

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            max_output_tokens=350,
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this vehicle damage:"},
                        {"type": "input_image", "image": image_b64},
                    ]
                }
            ]
        )
        return response.output_text

    except Exception as e:
        return f"AI model error: {e}"


# ---------------------------------------------------------
# Twilio Webhook Endpoint
# ---------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "")
    image_url = form.get("MediaUrl0")  # may be None

    reply = MessagingResponse()

    # If image received → perform AI estimate
    if image_url:
        try:
            image_b64 = download_and_encode(image_url)
            analysis = analyze_damage(image_b64)
            estimate_id = str(uuid.uuid4())[:12]

            reply.message(
                f"📘 AI Damage Estimate for {SHOP_NAME}\n\n"
                f"{analysis}\n\n"
                f"Estimate ID: {estimate_id}\n\n"
                "Reply with:\n"
                "1 — Book 9:00 AM\n"
                "2 — Book 11:00 AM\n"
                "3 — Book 2:00 PM\n"
            )
        except Exception as e:
            reply.message(f"❌ Error processing image: {e}")

        return Response(content=str(reply), media_type="application/xml")

    # If no image → ask user to send one
    reply.message(
        f"Welcome to {SHOP_NAME}.\n"
        "Please send a clear photo of the vehicle damage."
    )
    return Response(content=str(reply), media_type="application/xml")


# ---------------------------------------------------------
# Root health check
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "running", "shop": SHOP_NAME}


# ---------------------------------------------------------
# Railway startup
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        re
