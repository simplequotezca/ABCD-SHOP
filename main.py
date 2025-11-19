from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import os
import requests

app = FastAPI()

# ==========================
# CONFIG
# ==========================

SHOP_NAME = "Mississauga Collision Centre"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Allowed areas – ONLY these can ever appear in the final list
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
    "left rear tire", "right rear tire",
]

def clean_area_list(text: str):
    """
    Take raw model text and return ONLY allowed areas that are explicitly mentioned.
    No guessing. No extra areas.
    """
    text_lower = text.lower()
    found = []
    for area in ALLOWED_AREAS:
        if area in text_lower:
            found.append(area)
    # remove duplicates while keeping order
    unique = []
    for a in found:
        if a not in unique:
            unique.append(a)
    return unique


# ==========================
# PROMPTS
# ==========================

PRE_SCAN_PROMPT = """
You are an automotive AI DAMAGE PRE-SCAN system.

Your job is ONLY to visually scan the photo for CLEAR, OBVIOUS damage.

Rules:
1. ONLY list areas where visible damage is CLEAR in the image.
2. NEVER guess or mention areas that are not obviously damaged.
3. If the image is unclear, say so in NOTES.
4. Use simple bullet points.

STRICT output format:

AREAS:
- area 1
- area 2
- area 3

NOTES:
- short note here
"""

FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator in Ontario, Canada (year 2025).

You are given a list of CONFIRMED damaged areas from a pre-scan.
You MUST:
- ONLY use those confirmed areas.
- NOT add new areas.
- Be realistic and conservative with costs.

Output format:

AI Damage Estimate for {shop_name}

Severity: <Mild/Moderate/Severe>
Estimated Cost (Ontario 2025): $x – $y

Areas:
- area 1
- area 2
- area 3

Damage Types:
- type 1
- type 2
- type 3

Explanation:
<3–4 short lines explaining what likely needs to be repaired/replaced>
"""


# ==========================
# SIMPLE OPENAI HELPER
# ==========================

def openai_chat(messages, max_tokens: int = 500) -> str:
    """
    Call OpenAI Chat Completions API using HTTP requests.
    This avoids the python client issues and should be very stable.
    """
    if not OPENAI_API_KEY:
        # Don't crash the app if key is missing
        print("ERROR: OPENAI_API_KEY is not set")
        return "Internal configuration error (missing API key)."

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # Log error but don't crash
        print("Error calling OpenAI:", e)
        return "Sorry, our AI estimator is temporarily unavailable."


# ==========================
# VERY SIMPLE IN-MEMORY SESSION
# phone_number -> confirmed areas list
# ==========================

sessions = {}


# ==========================
# ROUTES
# ==========================

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "auto-shop-ai"}


async def handle_sms_request(form):
    body = (form.get("Body") or "").strip()
    body_lower = body.lower()
    from_number = form.get("From", "")
    num_media = int(form.get("NumMedia", "0") or "0")

    if not from_number:
        return PlainTextResponse("Error: no phone number provided.")

    # Start flow
    if body_lower in ("hi", "start"):
        msg = (
            f"Welcome to {SHOP_NAME} AI Damage Estimator.\n\n"
            "Step 1: Reply with a clear photo of the vehicle damage.\n"
            "Step 2: Confirm the areas I detect, and I'll send a detailed estimate."
        )
        return PlainTextResponse(msg)

    # If the user sent a photo -> PRE-SCAN
    if num_media > 0:
        media_url = form.get("MediaUrl0")
        if not media_url:
            return PlainTextResponse("I couldn't read the photo URL. Please resend the picture.")

        # Call OpenAI vision with remote image URL
        messages = [
            {
                "role": "system",
                "content": PRE_SCAN_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this vehicle damage photo. Follow the format exactly.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": media_url},
                    },
                ],
            },
        ]

        raw_text = openai_chat(messages, max_tokens=400)
        print("PRE-SCAN RAW:", raw_text)

        confirmed_areas = clean_area_list(raw_text)
        sessions[from_number] = confirmed_areas

        reply = f"AI Pre-Scan for {SHOP_NAME}\n\n"
        reply += "I can see possible damage in these areas:\n"
        if confirmed_areas:
            for a in confirmed_areas:
                reply += f"- {a}\n"
        else:
            reply += "- No clear damage detected from this photo.\n"

        reply += (
            "\nReply 1 if this looks roughly correct.\n"
            "Reply 2 if it's wrong or incomplete and you'll resend a clearer photo."
        )
        return PlainTextResponse(reply)

    # Step 2 – user confirms areas
    if body_lower == "1":
        confirmed_areas = sessions.get(from_number, [])
        if not confirmed_areas:
            return PlainTextResponse(
                "I don't have any saved damage areas for you yet. "
                "Please send a clear photo of the damage first."
            )

        system_prompt = FULL_ESTIMATE_PROMPT.format(shop_name=SHOP_NAME)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Confirmed damaged areas: {', '.join(confirmed_areas)}",
            },
        ]

        estimate_text = openai_chat(messages, max_tokens=450)
        return PlainTextResponse(estimate_text)

    # User says pre-scan is wrong
    if body_lower == "2":
        return PlainTextResponse(
            "No problem — please resend 1–3 clearer photos focused on the damaged area, "
            "and I'll rescan."
        )

    # Fallback
    return PlainTextResponse(
        "To start, send a clear photo of the vehicle damage.\n"
        "Or reply 'hi' to see instructions again."
    )


@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    return await handle_sms_request(form)


# Optional: second path for testing / manual requests
@app.post("/sms")
async def sms_compat(request: Request):
    form = await request.form()
    return await handle_sms_request(form)
