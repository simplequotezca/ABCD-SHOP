from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

# ---------------------------------------------------------
# STRICT PANEL VOCAB (prevents random / wrong areas)
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
    Take the AI text and extract ONLY areas from ALLOWED_AREAS
    that actually appear in the text.
    """
    text_lower = text.lower()
    found = []
    for area in ALLOWED_AREAS:
        if area in text_lower:
            found.append(area)
    # de-dupe but keep order
    unique = []
    for a in found:
        if a not in unique:
            unique.append(a)
    return unique


# ---------------------------------------------------------
# STEP 1: PRE-SCAN PROMPT (areas only, NO cost)
# ---------------------------------------------------------
PRE_SCAN_PROMPT = f"""
You are an automotive DAMAGE PRE-SCAN AI.

Your job:
- Look ONLY at the attached photo(s).
- Identify which exterior panels / areas have CLEAR VISIBLE damage.
- Use ONLY panel names from this list (exact wording):

{json.dumps(ALLOWED_AREAS, indent=2)}

Rules:
1. ONLY list areas where damage is clearly visible (dents, creases, scrapes, cracks, broken parts, obvious misalignment).
2. DO NOT guess about sides you cannot see.
3. DO NOT mention areas that look normal.
4. If you are unsure or the photo is too tight/blurred, keep AREAS empty and explain in NOTES.

Output format STRICTLY:

AREAS:
- panel name 1
- panel name 2

NOTES:
- short note about what you see / any uncertainty
"""


# ---------------------------------------------------------
# STEP 2: FULL ESTIMATE PROMPT (human style message)
# ---------------------------------------------------------
FULL_ESTIMATE_PROMPT = """
You are an AI collision estimator for Ontario, Canada (year 2025).

Input:
- A list of CONFIRMED damaged panels/areas (already verified by the customer).

Your job:
- Classify overall severity: Mild, Moderate, or Severe.
- Give a realistic Ontario 2025 cost RANGE (labour + materials) in CAD.
- List likely damage types in simple terms (scratches, dents, cracks, bumper deformation, curb rash, etc).
- Explain the estimate in 2–4 short friendly sentences.

Hard rules:
1. ONLY use the confirmed areas passed in by the user – do NOT add new panels.
2. Do NOT mention damage to areas that aren’t in the confirmed list.
3. Be realistic and conservative. This is a preliminary, visual-only estimate.

Output format EXACTLY:

AI Damage Estimate for {shop_name}

Severity: <Mild/Moderate/Severe>
Estimated Cost (Ontario 2025): $X – $Y

Areas:
- area 1
- area 2

Damage Types:
- type 1
- type 2

Explanation:
<3–4 short lines the shop can send directly to the customer>
"""


# ---------------------------------------------------------
# Simple in-memory session: phone -> confirmed areas
# ---------------------------------------------------------
sessions = {}  # { "+1xxx": {"areas": [...]} }


# ---------------------------------------------------------
# Twilio SMS webhook
# ---------------------------------------------------------
@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Twilio webhook:
    - User sends photo(s) -> we run PRE-SCAN, reply with areas, ask for 1/2.
    - User replies 1 -> run full estimate using ONLY those areas.
    - User replies 2 -> ask for clearer photos.
    - Any other text -> send instructions.
    """
    form = await request.form()
    body = (form.get("Body") or "").strip()
    lower_body = body.lower()
    from_number = form.get("From")
    num_media_str = form.get("NumMedia", "0") or "0"

    try:
        num_media = int(num_media_str)
    except ValueError:
        num_media = 0

    if not from_number:
        return PlainTextResponse("Error: missing phone number from request.")

    # -----------------------------------------------------
    # CASE 1: Incoming photo(s) -> run PRE-SCAN
    # -----------------------------------------------------
    if num_media > 0:
        image_urls = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            if url:
                image_urls.append(url)

        if not image_urls:
            return PlainTextResponse(
                "I couldn't read the photo. Please try again with at least one clear image of the damage."
            )

        # Build multimodal content: text + all images
        content = [
            {
                "type": "text",
                "text": (
                    "Analyze these vehicle photos ONLY for clearly visible exterior damage. "
                    "Follow the PRE-SCAN instructions and output format exactly."
                ),
            }
        ]
        for url in image_urls[:3]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": PRE_SCAN_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        pre_scan_text = completion.choices[0].message.content or ""
        confirmed_areas = clean_area_list(pre_scan_text)

        # Save to session for Step 2
        sessions[from_number] = {
            "areas": confirmed_areas,
        }

        # Build human-style Pre-Scan reply (like older version)
        lines = []
        lines.append("AI Pre-Scan for Mississauga Collision Centre")
        lines.append("")

        if confirmed_areas:
            lines.append("From these photo(s), I can clearly see damage in:")
            for a in confirmed_areas:
                lines.append(f"- {a}")
        else:
            lines.append(
                "I couldn't confidently lock onto specific damaged panels from this photo yet."
            )
            lines.append(
                "This is usually because the photo is too close, too dark, or only shows a tiny part of the vehicle."
            )

        lines.append("")
        lines.append("Reply 1 if this looks roughly correct.")
        lines.append("Reply 2 if it's wrong and you'll resend clearer photos.")

        return PlainTextResponse("\n".join(lines))

    # -----------------------------------------------------
    # CASE 2: User replies "1" -> full estimate
    # -----------------------------------------------------
    if lower_body in {"1", "yes", "y"}:
        session = sessions.get(from_number)
        if not session:
            return PlainTextResponse(
                "I don't see a recent photo from you. Please send 2–3 clear photos of the damaged area to start a new estimate."
            )

        confirmed_areas = session.get("areas") or []

        # Build user text for estimate
        area_lines = "\n".join(f"- {a}" for a in confirmed_areas) if confirmed_areas else "- (no clear areas)"
        user_text = (
            "These are the damaged areas the customer has confirmed as correct:\n"
            f"{area_lines}\n\n"
            "Based on this, create the estimate in the exact format you were given."
        )

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": FULL_ESTIMATE_PROMPT.format(
                        shop_name="Mississauga Collision Centre"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )

        estimate_text = completion.choices[0].message.content or ""

        # Clear session now that we've finished the flow
        sessions.pop(from_number, None)

        return PlainTextResponse(estimate_text)

    # -----------------------------------------------------
    # CASE 3: User replies "2" -> ask for clearer photos
    # -----------------------------------------------------
    if lower_body in {"2", "no", "n"}:
        sessions.pop(from_number, None)
        return PlainTextResponse(
            "No problem. Please send 2–3 clearer photos of the damage "
            "(one wider shot of the whole corner, plus 1–2 close-ups), and I'll scan it again."
        )

    # -----------------------------------------------------
    # CASE 4: Any other text -> instructions
    # -----------------------------------------------------
    instructions = (
        "Thanks for messaging Mississauga Collision Centre.\n\n"
        "To get an AI-powered pre-estimate:\n"
        "- Send 1–3 clear photos of the damaged area (front, rear, side, wheels, roof, etc.).\n"
        "- I'll scan the photos, list the damaged areas I see, and you can confirm.\n"
        "- After you reply 1 to confirm, I'll send a detailed Ontario 2025 cost estimate.\n"
    )
    return PlainTextResponse(instructions)


# ---------------------------------------------------------
# Simple health check
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI damage estimator is running"}
