from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import requests
import json

app = FastAPI()
client = OpenAI()

# ---------------------------------------------------------
# ALLOWED DAMAGE AREAS (STRICT FILTER TO STOP BAD OUTPUTS)
# ---------------------------------------------------------
ALLOWED_AREAS = [
    "front bumper upper", "front bumper lower", 
    "rear bumper upper", "rear bumper lower",
    "front left fender", "front right fender",
    "rear left fender", "rear right fender",
    "front left door", "front right door",
    "rear left door", "rear right door",
    "left quarter panel", "right quarter panel",
    "hood", "roof", "trunk", "tailgate",
    "windshield", "rear window", "left windows", "right windows",
    "left side mirror", "right side mirror",
    "left headlight", "right headlight",
    "left taillight", "right taillight",
    "left front wheel", "right front wheel",
    "left rear wheel", "right rear wheel",
    "left front tire", "right front tire",
    "left rear tire", "right rear tire"
]


def clean_area_list(text: str):
    """Find ONLY allowed areas inside the AI text."""
    text_lower = text.lower()
    found = [a for a in ALLOWED_AREAS if a in text_lower]
    return list(set(found))


# ---------------------------------------------------------
# PRE-SCAN PROMPT
# ---------------------------------------------------------
PRE_SCAN_PROMPT = """
You are an automotive DAMAGE PRE-SCAN AI.

Rules:
1. ONLY list areas where damage is clearly visible.
2. DO NOT guess.
3. DO NOT mention any areas not obviously damaged.
4. If unsure, write 'uncertain'.
5. Keep it short.

FORMAT STRICTLY:

AREAS:
- area1
- area2

NOTES:
- short note
"""


# ---------------------------------------------------------
# FULL ESTIMATE PROMPT
# ---------------------------------------------------------
FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator for Ontario (2025).

Rules:
1. ONLY use the areas CONFIRMED by the user.
2. Do NOT add new areas.
3. Provide:
   - Severity
   - Cost range (Ontario 2025)
   - Damage types
   - A simple explanation

FORMAT STRICTLY:

AI Damage Estimate for {shop_name}

Severity: <severity>
Estimated Cost (Ontario 2025): $X – $Y

Areas:
- area1
- area2

Damage Types:
- type1
- type2

Explanation:
3–4 short lines.
"""

# ---------------------------------------------------------
# SESSION MEMORY
# ---------------------------------------------------------
sessions = {}  # phone_number → confirmed_area_list


# ---------------------------------------------------------
# SMS WEBHOOK ROUTE
# ---------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):

    form = await request.form()
    body = (form.get("Body") or "").strip().lower()
    from_number = form.get("From")
    media_url = form.get("MediaUrl0")

    if not from_number:
        return PlainTextResponse("Error: Missing phone number.")

    # ---------------------------
    # START MESSAGE
    # ---------------------------
    if body in ["hi", "start"]:
        return PlainTextResponse(
            "Send me a clear photo of your vehicle damage to begin a Pre-Scan."
        )

    # ---------------------------
    # PRE-SCAN (PHOTO RECEIVED)
    # ---------------------------
    if media_url:
        print("Processing PRE-SCAN for:", media_url)

        # Call OpenAI "responses" API (correct 2025 format)
        pre_scan = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": PRE_SCAN_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this image for visible vehicle damage only."},
                        {"type": "input_image", "image_url": media_url}
                    ]
                }
            ]
        )

        text = pre_scan.output_text
        confirmed = clean_area_list(text)
        sessions[from_number] = confirmed

        reply = "AI Pre-Scan Results:\nI can clearly see damage in:\n"

        if confirmed:
            for a in confirmed:
                reply += f"- {a}\n"
        else:
            reply += "- No clearly visible damage detected\n"

        reply += (
            "\nReply 1 if this is correct.\n"
            "Reply 2 if wrong and you'll send a clearer photo."
        )

        return PlainTextResponse(reply)

    # ---------------------------
    # USER CONFIRMS → FINAL ESTIMATE
    # ---------------------------
    if body == "1":

        confirmed_areas = sessions.get(from_number, [])

        estimate = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": FULL_ESTIMATE_PROMPT.format(
                        shop_name="Mississauga Collision Centre"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Confirmed areas: {confirmed_areas}",
                },
            ],
        )

        return PlainTextResponse(estimate.output_text)

    # ---------------------------
    # USER REJECTS → RESEND PHOTO
    # ---------------------------
    if body == "2":
        return PlainTextResponse("Okay. Please send a clearer photo.")

    # ---------------------------
    # DEFAULT
    # ---------------------------
    return PlainTextResponse("Please send a vehicle damage photo to begin.")
