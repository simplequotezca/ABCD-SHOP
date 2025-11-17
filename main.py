import os
import uuid
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import datetime
import json

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
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


# ---------------------------------------------------------
# Generate Booking Options
# ---------------------------------------------------------
def generate_booking_slots():
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)

    slots = [
        tomorrow.replace(hour=9, minute=0),
        tomorrow.replace(hour=11, minute=0),
        tomorrow.replace(hour=14, minute=0),
    ]

    return [
        f"{slot.strftime('%a %b %d at %I:%M %p')}"
        for slot in slots
    ]


# ---------------------------------------------------------
# AI Damage Estimator — Ontario 2025 Calibration
# Structured JSON Output
# ---------------------------------------------------------
def analyze_damage(image_b64: str) -> dict:
    system_prompt = """
You are a professional 2025 Ontario (Canada) auto-body estimator
with 15+ years of experience.

Analyze the uploaded vehicle damage and RETURN ONLY JSON in this format:

{
  "severity": "Minor | Moderate | Severe | Total Loss",
  "damaged_panels": ["front bumper lower", "right fender", ...],
  "damage_types": ["dent", "deep scratch", ...],
  "recommended_repairs": ["PDR", "panel repair + paint", ...],
  "min_cost": 0,
  "max_cost": 0,
  "notes": "short explanation"
}

RULES:
- Use real Ontario 2025 market pricing.
- Be specific. NEVER say “general damage”.
- Identify actual panels shown.
- Identify exact repair operations.
- Convert everything into CAD.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        max_output_tokens=500,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this vehicle damage and return JSON only."},
                    {"type": "input_image", "image": image_b64},
                ]
            }
        ]
    )

    # Parse JSON
    try:
        text = response.output_text.strip()
        return json.loads(text)
    except Exception:
        return {
            "severity": "Moderate",
            "damaged_panels": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "notes": "AI fallback result"
        }


# ---------------------------------------------------------
# Twilio Webhook Endpoint
# ---------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    image_url = form.get("MediaUrl0")

    reply = MessagingResponse()

    # If image received → perform AI estimate
    if image_url:
        try:
            image_b64 = download_and_encode(image_url)
            result = analyze_damage(image_b64)
            estimate_id = str(uuid.uuid4())[:12]

            severity = result.get("severity")
            panels = ", ".join(result.get("damaged_panels", [])) or "None detected"
            types = ", ".join(result.get("damage_types", [])) or "None detected"
            repairs = ", ".join(result.get("recommended_repairs", [])) or "None"
            cost = f"${result.get('min_cost', 0):,.0f} - ${result.get('max_cost', 0):,.0f}"
            notes = result.get("notes", "")

            # Booking slots
            slots = generate_booking_slots()
            booking_text = "\n".join([f"{i+1} — {slot}" for i, slot in enumerate(slots)])

            reply.message(
                f"📘 AI Damage Estimate for {SHOP_NAME}\n"
                f"Severity: {severity}\n"
                f"Panels: {panels}\n"
                f"Damage Types: {types}\n"
                f"Repairs: {repairs}\n"
                f"Estimated Cost: {cost}\n"
                f"Notes: {notes}\n\n"
                f"Estimate ID: {estimate_id}\n\n"
                f"Reply with:\n{booking_text}"
            )

        except Exception as e:
            reply.message(f"❌ Error: {e}")

        return Response(content=str(reply), media_type="application/xml")

    # If no image → ask user to send one
    reply.message(
        f"Welcome to {SHOP_NAME}.\n"
        "Please send a clear photo of the vehicle damage to receive an AI estimate."
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
        reload=False
    )
