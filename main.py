from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import requests

app = FastAPI()

# OpenAI client will use OPENAI_API_KEY from your environment
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
    "left rear tire", "right rear tire",
]


def clean_area_list(text: str):
    """
    Scan the model's text and return only the allowed areas
    that actually appear in its output.
    """
    text_lower = text.lower()
    found = [a for a in ALLOWED_AREAS if a in text_lower]
    # Make unique
    return list(set(found))


# ---------------------------------------------------------
# PRE-SCAN PROMPT
# ---------------------------------------------------------
PRE_SCAN_PROMPT = """
You are an automotive DAMAGE PRE-SCAN AI.

Your ONLY job in this step:

- Look at the photo(s) of the vehicle.
- List which panels/areas appear clearly damaged.
- Be conservative and ONLY include areas where damage is obvious.
- Do NOT guess. If you're not sure, do NOT list it.
- Do NOT mention areas that are outside the frame.

Output format STRICTLY:

AREAS:
- area 1
- area 2
- area 3

NOTES:
- short note about what you see or any uncertainty
"""


# ---------------------------------------------------------
# FULL ESTIMATE PROMPT
# ---------------------------------------------------------
FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator for Ontario (2025).

Input:
- A list of CONFIRMED damaged areas that the customer has agreed to.

Rules:
1. ONLY use the confirmed areas. Do NOT add new areas.
2. Provide:
   - Severity (Minor / Moderate / Severe)
   - Cost range (Ontario 2025, in CAD)
   - Likely damage types (dent, scratch, crack, etc.)
   - A simple, friendly explanation for the customer.

You are writing an SMS-style message the shop will send to the customer.

Format your answer EXACTLY like this:

AI Damage Estimate for {shop_name}

Severity: <Minor/Moderate/Severe>
Estimated Cost (Ontario 2025): $X – $Y

Areas:
- area 1
- area 2

Damage Types:
- type 1
- type 2

Explanation:
3–4 short lines in plain language.
"""


# ---------------------------------------------------------
# SESSION MEMORY (very simple, in-memory)
# ---------------------------------------------------------
# from_number -> confirmed_area_list
sessions = {}


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI damage estimator is running"}


# ---------------------------------------------------------
# TWILIO SMS WEBHOOK ROUTE
# ---------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio will POST here:
    https://web-production-a1388.up.railway.app/sms-webhook

    Flow:
      - Customer sends photo(s) -> we run PRE-SCAN and ask "Reply 1 or 2".
      - Customer replies "1" -> we run full estimate and reply with cost.
      - Customer replies "2" -> we ask them to resend clearer photos.
    """
    form = await request.form()
    body = (form.get("Body") or "").strip().lower()
    from_number = form.get("From")
    media_url = form.get("MediaUrl0")

    if not from_number:
        return PlainTextResponse("Error: Missing phone number.")

    # -----------------------------------------------------
    # START: if they type hi / start, explain how to use it
    # -----------------------------------------------------
    if body in ["hi", "start"]:
        return PlainTextResponse(
            "Thanks for contacting the AI damage estimator.\n\n"
            "To begin, please send 1–3 clear photos of the vehicle damage."
        )

    # -----------------------------------------------------
    # STEP 1: PRE-SCAN (PHOTO RECEIVED)
    # -----------------------------------------------------
    if media_url:
        # (Optional) Download to make sure URL works
        # If Twilio's URL is public, OpenAI can fetch it directly via image_url.
        try:
            _ = requests.get(media_url, timeout=10)
        except Exception:
            # If download fails, still try giving URL to OpenAI – often still works
            pass

        # Call OpenAI Chat Completions with vision
        pre_scan = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": PRE_SCAN_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image for clearly visible vehicle damage only. "
                                    "Follow the output format exactly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": media_url},
                        },
                    ],
                },
            ],
        )

        text = pre_scan.choices[0].message.content or ""
        confirmed = clean_area_list(text)
        sessions[from_number] = confirmed

        # Build human-friendly SMS
        reply = "AI Pre-Scan Results:\n"

        if confirmed:
            reply += "I can clearly see damage in:\n"
            for a in confirmed:
                reply += f"- {a}\n"
        else:
            reply += (
                "I couldn't clearly identify specific damaged panels from this photo.\n"
            )

        reply += (
            "\nReply 1 if this looks roughly correct.\n"
            "Reply 2 if it's wrong and you'll send a clearer photo."
        )

        return PlainTextResponse(reply)

    # -----------------------------------------------------
    # STEP 2: USER CONFIRMS (body == "1")
    # -----------------------------------------------------
    if body == "1":
        confirmed_areas = sessions.get(from_number, [])

        # If we have no stored areas, ask for a new photo
        if not confirmed_areas:
            return PlainTextResponse(
                "I don't see a recent photo for your number.\n"
                "Please send 1–3 clear photos of the vehicle damage to start a new estimate."
            )

        full_estimate = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": FULL_ESTIMATE_PROMPT.format(
                        shop_name="Mississauga Collision Centre"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The customer has confirmed these damaged areas:\n"
                        + "\n".join(f"- {a}" for a in confirmed_areas)
                    ),
                },
            ],
        )

        output = full_estimate.choices[0].message.content or ""
        # (optional) clear session after final estimate
        sessions.pop(from_number, None)
        return PlainTextResponse(output)

    # -----------------------------------------------------
    # STEP 2: USER REJECTS (body == "2")
    # -----------------------------------------------------
    if body == "2":
        sessions.pop(from_number, None)
        return PlainTextResponse(
            "No problem.\n"
            "Please send 2–3 clearer photos of the damage "
            "(wide shot of the corner plus a couple of close-ups) and I'll rescan it."
        )

    # -----------------------------------------------------
    # DEFAULT: No photo + not 1/2 -> send instructions
    # -----------------------------------------------------
    return PlainTextResponse(
        "To get an AI damage estimate:\n"
        "1) Send 1–3 clear photos of the damaged area.\n"
        "2) I'll send a pre-scan of what I see.\n"
        "3) Reply 1 if it's correct, or 2 if you'll resend photos.\n"
    )
