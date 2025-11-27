import os
import json
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from google.oauth2 import service_account
from googleapiclient.discovery import build

# ============================================================
# ENV + BASIC CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

SHOPS_JSON = os.getenv("SHOPS_JSON")
if not SHOPS_JSON:
    raise RuntimeError(
        "SHOPS_JSON env var is required. Example:\n"
        '[{"id":"miss","name":"Mississauga Collision Centre","webhook_token":"shop_miss_123",'
        '"calendar_id":"shiran.bookings@gmail.com","pricing":{...},"hours":{...}}]'
    )

try:
    SHOPS = json.loads(SHOPS_JSON)
except Exception as e:
    raise RuntimeError(f"Failed to parse SHOPS_JSON: {e}")

SHOP_BY_TOKEN: Dict[str, Dict[str, Any]] = {
    shop["webhook_token"]: shop for shop in SHOPS
}

# Timezone used for appointments (matches your shop in Ontario)
LOCAL_TZ = "America/Toronto"

# In-memory conversation store (per phone number + shop)
IN_MEMORY_CONVOS: Dict[str, Dict[str, Any]] = {}
CONVO_TTL_SECONDS = 60 * 60 * 2  # 2 hours


app = FastAPI()


# ============================================================
# HELPERS – CONVERSATIONS
# ============================================================

def convo_key(shop_id: str, phone: str) -> str:
    return f"{shop_id}:{phone}"


def get_convo(shop_id: str, phone: str) -> Optional[Dict[str, Any]]:
    key = convo_key(shop_id, phone)
    convo = IN_MEMORY_CONVOS.get(key)
    if not convo:
        return None
    # TTL cleanup
    if (datetime.utcnow() - convo["updated_at"]).total_seconds() > CONVO_TTL_SECONDS:
        del IN_MEMORY_CONVOS[key]
        return None
    return convo


def save_convo(shop_id: str, phone: str, data: Dict[str, Any]) -> None:
    key = convo_key(shop_id, phone)
    data["updated_at"] = datetime.utcnow()
    IN_MEMORY_CONVOS[key] = data


def clear_convo(shop_id: str, phone: str) -> None:
    key = convo_key(shop_id, phone)
    IN_MEMORY_CONVOS.pop(key, None)


# ============================================================
# HELPERS – OPENAI (HTTP DIRECT, NO SDK)
# ============================================================

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def call_openai_chat(
    model: str,
    messages: list,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> str:
    """
    Minimal wrapper around OpenAI Chat Completions API.
    We do NOT use the Python openai SDK so we never hit the 'proxies' bug.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Extract the first {...} JSON block from model output.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    raw = text[start : end + 1]
    return json.loads(raw)


# ============================================================
# HELPERS – GOOGLE CALENDAR
# ============================================================

def get_calendar_service():
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON env var is required for calendar integration.\n"
            "Paste your service account JSON as a single line."
        )
    info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    service = build("calendar", "v3", credentials=creds)
    return service


def create_calendar_appointment(
    shop: Dict[str, Any],
    customer_name: str,
    customer_phone: str,
    customer_email: str,
    vehicle_info: str,
    est_data: Dict[str, Any],
    start_iso_local: str,
) -> str:
    """
    Booking option 2: customer chooses preferred day/time.
    Event style 2: clean title, full details in description.
    """
    service = get_calendar_service()

    # Start time in local ISO like '2025-11-28T15:30'
    start_dt = datetime.fromisoformat(start_iso_local)
    end_dt = start_dt + timedelta(hours=1)

    summary = f"Repair appointment – {customer_name}"

    description_lines = [
        f"Customer: {customer_name}",
        f"Phone: {customer_phone}",
        f"Email: {customer_email or 'N/A'}",
        "",
        f"Vehicle: {vehicle_info or 'N/A'}",
        "",
        f"Shop: {shop['name']}",
        "",
        "AI Damage Estimate:",
        f"- Severity: {est_data.get('severity', 'N/A')}",
        f"- Estimated Cost: {est_data.get('estimated_cost_range', 'N/A')}",
        f"- Repair Time: {est_data.get('repair_time', 'N/A')}",
        "",
        "Explanation:",
        est_data.get("explanation", ""),
        "",
        "Safety notes:",
        est_data.get("safety_notes", ""),
    ]

    event_body = {
        "summary": summary,
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": LOCAL_TZ,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": LOCAL_TZ,
        },
    }

    event = (
        service.events()
        .insert(calendarId=shop["calendar_id"], body=event_body, sendUpdates="all")
        .execute()
    )
    return event.get("htmlLink", "")


# ============================================================
# HELPERS – HOURS + PRICING
# ============================================================

def format_shop_hours(hours_cfg: Dict[str, Any]) -> str:
    # Simple human readable hours line
    order = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    parts = []
    for d in order:
        slots = hours_cfg.get(d, [])
        disp_day = d.capitalize()
        if not slots or "closed" in [s.lower() for s in slots]:
            parts.append(f"{disp_day}: closed")
        else:
            parts.append(f"{disp_day}: {', '.join(slots)}")
    return " | ".join(parts)


def build_estimate_sms(shop: Dict[str, Any], est: Dict[str, Any]) -> str:
    pricing = shop.get("pricing", {})
    labor = pricing.get("labor_rates", {})
    body_rate = labor.get("body")
    paint_rate = labor.get("paint")
    materials_rate = pricing.get("materials_rate")

    lines = [
        f"{shop['name']} – AI Damage Estimate",
        "",
        f"Severity: {est.get('severity', 'N/A').title()}",
        f"Estimated cost: {est.get('estimated_cost_range', 'N/A')}",
        f"Repair time: {est.get('repair_time', 'N/A')}",
        "",
        est.get("explanation", "").strip(),
    ]

    safety = est.get("safety_notes", "")
    if safety:
        lines.append("")
        lines.append(f"Safety notes: {safety}")

    if body_rate and paint_rate and materials_rate:
        lines.append("")
        lines.append(
            f"(Based on approx. ${body_rate}/hr body, ${paint_rate}/hr paint "
            f"and ${materials_rate}/hr materials at our shop.)"
        )

    lines.append("")
    lines.append(
        "This is a visual preliminary estimate, not a final repair bill."
    )

    return "\n".join(lines)


# ============================================================
# AI DAMAGE ESTIMATE (Pricing mode B: dynamic within floors)
# ============================================================

async def ai_damage_estimate(
    shop: Dict[str, Any],
    media_urls: list,
    user_text: str,
) -> Dict[str, Any]:
    """
    - Uses images + optional text description.
    - Uses shop.pricing + shop.hours.
    - Returns structured JSON with severity, cost range, etc.
    Pricing mode B = dynamic: model picks a range within floors.
    """

    pricing = shop.get("pricing", {})
    labor_rates = pricing.get("labor_rates", {})
    base_floor = pricing.get("base_floor", {})
    materials_rate = pricing.get("materials_rate")

    hours_str = format_shop_hours(shop.get("hours", {}))

    system_message = {
        "role": "system",
        "content": (
            "You are an Ontario (Canada, 2025) auto body damage estimator for a collision centre.\n"
            "Your job:\n"
            "1) Look at the damage photos and short description.\n"
            "2) Classify severity into one of: minor, moderate, severe.\n"
            "3) Choose an estimated cost RANGE in CAD that matches severity.\n"
            "4) Use the shop's pricing floors as soft boundaries.\n"
            "5) Explain the reasoning in plain language.\n\n"
            "Shop pricing info:\n"
            f"- Body labor rate: ${labor_rates.get('body', 'N/A')}/hr\n"
            f"- Paint labor rate: ${labor_rates.get('paint', 'N/A')}/hr\n"
            f"- Materials rate: ${materials_rate}/hr\n"
            "Price floors (CAD):\n"
            f"- Minor: {base_floor.get('minor_min')} – {base_floor.get('minor_max')}\n"
            f"- Moderate: {base_floor.get('moderate_min')} – {base_floor.get('moderate_max')}\n"
            f"- Severe: {base_floor.get('severe_min')} – {base_floor.get('severe_max')}\n\n"
            "Rules:\n"
            "- Pick a realistic range that reflects complexity & number of panels.\n"
            "- Stay within the appropriate floor range for the chosen severity.\n"
            "- If unsure between two severities, pick the lower severity but widen the range.\n"
            "- NEVER promise an exact price; always present a range.\n\n"
            "Output ONLY JSON in this exact schema:\n"
            "{\n"
            '  \"severity\": \"minor\" | \"moderate\" | \"severe\",\n'
            '  \"estimated_cost_range\": \"CAD $LOW – $HIGH\",\n'
            '  \"repair_time\": \"short human description like 1–3 business days\",\n'
            '  \"explanation\": \"1–3 sentences explaining what you see and what work is likely needed\",\n'
            '  \"safety_notes\": \"Any safety concerns (e.g., lights, structure) or \"\" if none\"\n'
            "}"
        ),
    }

    user_contents = []
    # description text
    if user_text:
        user_contents.append(
            {"type": "text", "text": f"Customer description: {user_text}"}
        )
    # add each image
    for url in media_urls:
        user_contents.append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )

    user_message = {
        "role": "user",
        "content": user_contents,
    }

    raw = call_openai_chat(
        model="gpt-4.1-mini",
        messages=[system_message, user_message],
        temperature=0.25,
        max_tokens=600,
    )
    data = extract_json_block(raw)

    # Quick sanity clamp: keep range within floors
    try:
        floors = base_floor or {}
        est_range_str = data.get("estimated_cost_range", "")
        # Very naive number pull
        nums = [
            float("".join(ch for ch in token if ch.isdigit() or ch == "."))
            for token in est_range_str.replace("$", "").replace(",", "").split()
            if any(c.isdigit() for c in token)
        ]
        if len(nums) >= 2:
            lo, hi = min(nums), max(nums)
            severity = data.get("severity", "moderate").lower()
            floor_lo = float(floors.get(f"{severity}_min") or lo)
            floor_hi = float(floors.get(f"{severity}_max") or hi)

            lo = max(lo, floor_lo)
            hi = min(hi, floor_hi)
            if hi < lo:
                hi = lo + 200  # simple fallback

            data["estimated_cost_range"] = f"CAD ${int(lo):,} – ${int(hi):,}"
    except Exception:
        # If anything fails, just keep model's original string
        pass

    return data


# ============================================================
# AI – PARSE APPOINTMENT DETAILS FROM FINAL TEXT
# ============================================================

def ai_parse_appointment(
    shop: Dict[str, Any],
    name: str,
    phone: str,
    email: str,
    vehicle: str,
    preferred_text: str,
) -> Dict[str, Any]:
    """
    Booking flow 2: customer picks their preferred day & time.
    We ask the model to convert free-form text into a strict ISO datetime.
    """
    hours_str = format_shop_hours(shop.get("hours", {}))

    system_message = {
        "role": "system",
        "content": (
            "You are scheduling a repair appointment for an auto body shop in the "
            f"{LOCAL_TZ} timezone.\n"
            "Customer has already given name, email, phone, vehicle details.\n"
            "Now they wrote a free-form message with their preferred day/time.\n\n"
            f"Shop hours: {hours_str}\n\n"
            "Your job: choose a realistic appointment datetime that falls within shop hours.\n"
            "If the text is vague like 'tomorrow morning', pick a specific reasonable time.\n"
            "Prefer the next available matching date/time from today onwards.\n\n"
            "Output ONLY JSON in this schema:\n"
            "{\n"
            '  \"appointment_start_iso\": \"YYYY-MM-DDTHH:MM\"  // local shop time,\n'
            '  \"notes\": \"Short note about how you interpreted their message\"\n'
            "}"
        ),
    }

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Customer info:\n"
                    f"- Name: {name}\n"
                    f"- Phone: {phone}\n"
                    f"- Email: {email}\n"
                    f"- Vehicle: {vehicle}\n\n"
                    f"Preferred time message: {preferred_text}"
                ),
            }
        ],
    }

    raw = call_openai_chat(
        model="gpt-4.1-mini",
        messages=[system_message, user_message],
        temperature=0.15,
        max_tokens=400,
    )
    return extract_json_block(raw)


# ============================================================
# MAIN TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    """
    Core Twilio SMS handler.
    URL should be: /sms-webhook?token=shop_miss_123
    """
    token = request.query_params.get("token")
    shop = SHOP_BY_TOKEN.get(token)
    if not shop:
        # Unknown shop – return simple plain response so Twilio isn't angry
        return PlainTextResponse("Unknown shop token", status_code=400)

    form = await request.form()
    from_number = form.get("From", "")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", "0") or "0")

    media_urls = []
    for i in range(num_media):
        url_key = f"MediaUrl{i}"
        url = form.get(url_key)
        if url:
            media_urls.append(url)

    resp = MessagingResponse()
    msg = resp.message()

    convo = get_convo(shop["id"], from_number)

    # ========================================================
    # CASE 1 – We are in the middle of the booking flow
    # ========================================================
    if convo:
        stage = convo.get("stage")

        # Customer can always type 'restart' to start over
        if body.lower() in {"restart", "start over", "new"}:
            clear_convo(shop["id"], from_number)
            msg.body(
                "No problem – let's start a new estimate.\n\n"
                "Please reply with clear photos of the damage (different angles). "
                "You can also add a short description if you like."
            )
            return Response(content=str(resp), media_type="application/xml")

        # Stage: we already sent estimate and asked for name
        if stage == "get_name":
            convo["name"] = body
            convo["stage"] = "get_email"
            save_convo(shop["id"], from_number, convo)
            msg.body(
                f"Thanks {body}! What's the best email address to send your estimate & booking confirmation?"
            )
            return Response(content=str(resp), media_type="application/xml")

        if stage == "get_email":
            convo["email"] = body
            convo["stage"] = "get_vehicle"
            save_convo(shop["id"], from_number, convo)
            msg.body(
                "Got it. What vehicle are we working on? (Year, make, model, colour)."
            )
            return Response(content=str(resp), media_type="application/xml")

        if stage == "get_vehicle":
            convo["vehicle"] = body
            convo["stage"] = "get_datetime"
            save_convo(shop["id"], from_number, convo)
            hours_str = format_shop_hours(shop.get("hours", {}))
            msg.body(
                "Perfect. Lastly, what day and time works best for you to bring the vehicle in?\n\n"
                f"Our hours: {hours_str}\n\n"
                "You can reply with something like 'next Tuesday at 3pm' or "
                "'tomorrow morning around 10'."
            )
            return Response(content=str(resp), media_type="application/xml")

        if stage == "get_datetime":
            preferred_text = body
            name = convo.get("name", "Customer")
            email = convo.get("email", "")
            vehicle = convo.get("vehicle", "")
            est_data = convo.get("estimate", {})

            try:
                parsed = ai_parse_appointment(
                    shop=shop,
                    name=name,
                    phone=from_number,
                    email=email,
                    vehicle=vehicle,
                    preferred_text=preferred_text,
                )
                start_iso = parsed.get("appointment_start_iso")
                if not start_iso:
                    raise ValueError("Missing appointment_start_iso")

                event_link = create_calendar_appointment(
                    shop=shop,
                    customer_name=name,
                    customer_phone=from_number,
                    customer_email=email,
                    vehicle_info=vehicle,
                    est_data=est_data,
                    start_iso_local=start_iso,
                )

                clear_convo(shop["id"], from_number)

                confirmation_lines = [
                    f"You're booked in, {name}! ✅",
                    "",
                    f"Date & time: {start_iso.replace('T', ' ')} ({LOCAL_TZ})",
                    f"Location: {shop['name']}",
                ]
                if event_link:
                    confirmation_lines.append("")
                    confirmation_lines.append(
                        "Calendar link:\n" + event_link
                    )
                confirmation_lines.append("")
                confirmation_lines.append(
                    "If you need to change anything, just reply to this message."
                )

                msg.body("\n".join(confirmation_lines))
                return Response(content=str(resp), media_type="application/xml")

            except Exception:
                # If anything fails, we keep it graceful
                msg.body(
                    "I had trouble scheduling that time automatically.\n\n"
                    "Please reply with a specific date & time in this format, "
                    "for example: 2025-12-01 15:30"
                )
                return Response(content=str(resp), media_type="application/xml")

        # Fallback for unknown stage – reset
        clear_convo(shop["id"], from_number)
        msg.body(
            "Let's restart your estimate.\n\n"
            "Please send clear photos of the damage (different angles), "
            "and a short description if you like."
        )
        return Response(content=str(resp), media_type="application/xml")

    # ========================================================
    # CASE 2 – No active conversation → Expect images & run estimate
    # ========================================================
    if not media_urls:
        # No images – explain what to do
        msg.body(
            f"Welcome to {shop['name']} AI estimate assistant.\n\n"
            "Please reply with clear photos of the damage (different angles). "
            "You can also include a short description in the same message."
        )
        return Response(content=str(resp), media_type="application/xml")

    try:
        est = await ai_damage_estimate(
            shop=shop,
            media_urls=media_urls,
            user_text=body,
        )
        est_sms = build_estimate_sms(shop, est)

        # Start a new conversation – pricing mode B, booking flow 2
        convo = {
            "stage": "get_name",
            "estimate": est,
        }
        save_convo(shop["id"], from_number, convo)

        hours_str = format_shop_hours(shop.get("hours", {}))

        msg.body(
            est_sms
            + "\n\n"
            "If you'd like to book an appointment, let's grab a few details.\n"
            "1) What's your full name?\n\n"
            f"(Shop hours for booking: {hours_str})"
        )

        return Response(content=str(resp), media_type="application/xml")

    except Exception as e:
        # Log error to container logs
        print("Error in ai_damage_estimate:", repr(e))
        msg.body(
            "Sorry, something went wrong generating the AI estimate.\n"
            "Please call the shop directly and we can help you over the phone."
        )
        return Response(content=str(resp), media_type="application/xml")


@app.get("/")
async def health():
    return {"status": "ok", "message": "Body shop AI estimator running"}
