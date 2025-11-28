import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel

# OpenAI
from openai import OpenAI

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI()


# ============================================================
# OpenAI setup
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# Shop config via SHOPS_JSON
# ============================================================

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str        # ?token=...
    calendar_id: Optional[str] = None
    timezone: Optional[str] = "America/Toronto"


def load_shops() -> Dict[str, Shop]:
    """
    SHOPS_JSON example:

    [
      {
        "id": "miss",
        "name": "Mississauga Collision Centre",
        "webhook_token": "shop_miss_123",
        "calendar_id": "shiran.bookings@gmail.com"
      }
    ]
    """
    raw = os.getenv("SHOPS_JSON")
    mapping: Dict[str, Shop] = {}

    if raw:
        try:
            data = json.loads(raw)
            for item in data:
                shop = Shop(**item)
                mapping[shop.webhook_token] = shop
        except Exception as e:
            print("Failed to parse SHOPS_JSON:", e)

    # Fallback single test shop (only if SHOPS_JSON missing/broken)
    if not mapping:
        default = Shop(
            id="default",
            name="Default Collision Centre",
            webhook_token="test_token",
            calendar_id=None,
        )
        mapping[default.webhook_token] = default

    return mapping


SHOPS_BY_TOKEN: Dict[str, Shop] = load_shops()


def get_shop_from_token(token: Optional[str]) -> Shop:
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=400, detail="Invalid or missing shop token")
    return SHOPS_BY_TOKEN[token]


# ============================================================
# Google Calendar client (optional)
# ============================================================

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def build_calendar_service():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        print("GOOGLE_SERVICE_ACCOUNT_JSON not set – calendar integration disabled.")
        return None
    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
        service = build("calendar", "v3", credentials=creds)
        return service
    except Exception as e:
        print("Failed to init Google Calendar service:", e)
        return None


calendar_service = build_calendar_service()


def maybe_create_calendar_event(
    shop: Shop,
    sms_body: str,
    from_number: str,
    estimate_text: str,
) -> Optional[str]:
    """
    Very simple auto-booking:
    If SMS body contains the word 'book', we try to create a 1-hour slot.

    If we can parse a date/time like 2025-12-01 14:30 we use that.
    Otherwise we default to tomorrow at 10:00.
    """
    if not calendar_service or not shop.calendar_id:
        return None

    if "book" not in sms_body.lower():
        return None

    tz = shop.timezone or "America/Toronto"

    # Try to find explicit date/time; if missing, fallback to tomorrow at 10:00.
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", sms_body)
    time_match = re.search(r"(\d{1,2}:\d{2})", sms_body)

    start: datetime
    if date_match and time_match:
        when_str = f"{date_match.group(1)} {time_match.group(1)}"
        try:
            start = datetime.fromisoformat(when_str)
        except Exception:
            start = datetime.now() + timedelta(days=1)
            start = start.replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        start = datetime.now() + timedelta(days=1)
        start = start.replace(hour=10, minute=0, second=0, microsecond=0)

    end = start + timedelta(hours=1)

    event_body = {
        "summary": f"AI Estimate Customer - {from_number}",
        "description": (
            f"Customer phone: {from_number}\n\n"
            f"Last SMS:\n{sms_body}\n\n"
            f"AI Estimate:\n{estimate_text}"
        ),
        "start": {"dateTime": start.isoformat(), "timeZone": tz},
        "end": {"dateTime": end.isoformat(), "timeZone": tz},
    }

    try:
        event = calendar_service.events().insert(
            calendarId=shop.calendar_id, body=event_body
        ).execute()
        return event.get("htmlLink")
    except Exception as e:
        print("Google Calendar error:", e)
        return None


# ============================================================
# AI estimator
# ============================================================

async def generate_ai_estimate(
    shop: Shop,
    from_number: str,
    body: str,
    image_url: Optional[str],
) -> str:
    """
    Uses OpenAI to create a structured repair estimate.
    If AI fails for any reason, returns a friendly fallback message.
    """

    system_prompt = (
        f"You are an expert auto body damage estimator for {shop.name} in Ontario, "
        "Canada (year 2025). You analyse photos and descriptions of vehicle damage "
        "and produce realistic, insurance-style preliminary estimates. "
        "Return clear, concise results that a customer can understand."
    )

    user_content: List[Dict] = []

    text_block = (
        "Customer phone: " + from_number + "\n\n"
        "Customer message:\n" + (body or "(no text message)") + "\n\n"
        "TASK:\n"
        "- If a photo is provided, carefully inspect damage.\n"
        "- Classify severity: minor / moderate / severe.\n"
        "- Provide a cost RANGE in CAD for Ontario 2025 prices.\n"
        "- List main areas and damage types.\n"
        "- Add a short explanation.\n\n"
        "FORMAT EXACTLY AS:\n"
        "Severity: <minor/moderate/severe>\n"
        "Estimated Cost (Ontario 2025): $X – $Y\n"
        "Areas: <comma-separated>\n"
        "Damage Types: <comma-separated>\n"
        "Short Explanation: <2–4 sentences>\n"
    )

    user_content.append({"type": "text", "text": text_block})

    if image_url:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=700,
        )
        content = completion.choices[0].message.content
        return content
    except Exception as e:
        print("OpenAI error:", e)
        return (
            "We tried to run the AI estimator but hit a temporary issue. "
            "Your photos and details have been received. A team member will "
            "review your damage and follow up with an estimate shortly."
        )


# ============================================================
# FastAPI routes
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI estimator + Google Calendar webhook is running.",
    }


@app.post("/sms-webhook", response_class=PlainTextResponse)
async def sms_webhook(request: Request):
    """
    Twilio SMS/MMS webhook.
    URL format: https://<your-app>.up.railway.app/sms-webhook?token=shop_miss_123
    """
    token = request.query_params.get("token")
    shop = get_shop_from_token(token)

    form = await request.form()

    from_number = form.get("From", "Unknown")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia") or 0)
    image_url = form.get("MediaUrl0") if num_media > 0 else None

    print(
        f"[SMS] shop={shop.id} from={from_number} "
        f"body={body!r} image_url={image_url!r}"
    )

    resp = MessagingResponse()
    message = resp.message()

    # --- 1) Always try AI estimator (handles text-only OR photo) ---
    estimate_text = await generate_ai_estimate(
        shop=shop,
        from_number=from_number,
        body=body,
        image_url=image_url,
    )

    # --- 2) Optional Calendar booking if they say 'book' ---
    calendar_link = maybe_create_calendar_event(
        shop=shop,
        sms_body=body,
        from_number=from_number,
        estimate_text=estimate_text,
    )

    # --- 3) Build final SMS reply ---
    if calendar_link:
        reply_body = (
            f"{estimate_text}\n\n"
            "Your requested time has been added to our calendar. "
            "If you need to change anything, please call the shop directly.\n\n"
            f"Calendar confirmation: {calendar_link}"
        )
    else:
        reply_body = (
            f"{estimate_text}\n\n"
            "To book an appointment, reply with the word 'book' plus your preferred "
            "date & time (e.g. 'book 2025-12-01 10:30') and your full name & email. "
            "We'll reserve the closest available spot in our calendar."
        )

    message.body(reply_body)

    # Twilio expects raw XML back
    return PlainTextResponse(str(resp))
