from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel
import json
import uuid
import requests

app = FastAPI()
client = OpenAI()

# -----------------------
# SAFETY / VALIDATION RULES
# -----------------------
ALLOWED_AREAS = [
    "front bumper upper", "front bumper lower", "rear bumper upper", "rear bumper lower",
    "front left fender", "front right fender", "rear left fender", "rear right fender",
    "front left door", "front right door", "rear left door", "rear right door",
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

def clean_area_list(text):
    final = []
    for a in ALLOWED_AREAS:
        if a in text.lower():
            final.append(a)
    return list(set(final))


# -----------------------
# STEP 1: PRE-SCAN PROMPT
# -----------------------
PRE_SCAN_PROMPT = """
You are an automotive AI DAMAGE PRE-SCAN system.

Rules:
1. ONLY list areas where visible damage is CLEAR.
2. NEVER guess or mention areas that are not obviously damaged.
3. Use simple bullet points.
4. If unsure, say "uncertain".

Output format strictly:
AREAS:
- area 1
- area 2
- area 3

NOTES:
- short note here
"""

# -----------------------
# STEP 2: FULL ESTIMATE PROMPT
# -----------------------
FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator for Ontario, Canada (2025).
You must be extremely accurate and conservative.

Rules:
1. ONLY use the confirmed areas from the user.
2. DO NOT add new areas not confirmed.
3. Provide:
   - Severity
   - Cost range (Ontario 2025)
   - Damage types
   - Short explanation
4. Be realistic. No exaggeration.

Format:
AI Damage Estimate for {shop_name}

Severity: <Mild/Moderate/Severe>
Estimated Cost (Ontario 2025): $x – $y

Areas:
- area 1
- area 2

Damage Types:
- type 1
- type 2

Explanation:
<3–4 lines>
"""

# -----------------------
# STEP HANDLING
# -----------------------
sessions = {}  # phone → stored areas


@app.post("/sms")
async def sms_webhook(request: Request):
    form = await request.form()
    body = form.get("Body", "").strip().lower()
    from_number = form.get("From")
    media_url = form.get("MediaUrl0")

    if not from_number:
        return PlainTextResponse("Error: No phone number received")

    # Start new thread
    if body == "hi" or body == "start":
        return PlainTextResponse(
            "Send me a photo of your vehicle damage to begin a Pre-Scan."
        )

    # Photo received → PRE-SCAN
    if media_url:
        img_bytes = requests.get(media_url).content

        pre_scan = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PRE_SCAN_PROMPT},
                {"role": "user", "content": "Analyze this image for damage."},
                {"role": "user", "content": [{"type": "image_url", "image_url": media_url}]}
            ]
        )

        text = pre_scan.choices[0].message.content

        # Extract allowed areas only
        confirmed = clean_area_list(text)
        sessions[from_number] = confirmed

        reply = "AI Pre-Scan:\nI can see damage in these areas:\n"
        if confirmed:
            for c in confirmed:
                reply += f"- {c}\n"
        else:
            reply += "- No clear damage detected\n"

        reply += "\nReply 1 if this is correct.\nReply 2 to resend a clearer photo."

        return PlainTextResponse(reply)

    # Step 2 confirmation:
    if body == "1":
        confirmed_areas = sessions.get(from_number, [])

        full_estimate = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FULL_ESTIMATE_PROMPT.format(shop_name="Mississauga Collision Centre")},
                {"role": "user", "content": f"Confirmed areas: {confirmed_areas}"}
            ]
        )

        output = full_estimate.choices[0].message.content
        return PlainTextResponse(output)

    if body == "2":
        return PlainTextResponse("Okay — please resend a clearer photo.")

    return PlainTextResponse("Please send a vehicle damage photo to begin.")
