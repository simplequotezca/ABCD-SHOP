import os
import json
import base64
import uuid
import logging
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from openai import OpenAI
from sqlalchemy import create_engine, text

# Optional Google Calendar (fails safe if not configured)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except ImportError:  # if libs missing it will just disable calendar
    service_account = None
    build = None

# ============================================================
# Basic setup
# ============================================================

app = FastAPI()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SHOPS_JSON = os.getenv("SHOPS_JSON")  # list of shops with webhook_token, name, etc.

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN env vars are required")

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ============================================================
# DB: make sure estimates table exists (minimal schema)
# ============================================================

def init_db():
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS estimates (
                    id TEXT PRIMARY KEY,
                    phone TEXT,
                    shop_id TEXT,
                    image_url TEXT,
                    ai_analysis TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        )


init_db()

# ============================================================
# Shops loader
# ============================================================


def load_shops_by_token():
    """Returns dict mapping webhook_token -> shop config."""
    if not SHOPS_JSON:
        return {}
    try:
        data = json.loads(SHOPS_JSON)
        by_token = {}
        for shop in data:
            token = shop.get("webhook_token")
            if token:
                by_token[token] = shop
        return by_token
    except Exception as e:
        logging.error(f"Failed to parse SHOPS_JSON: {e}")
        return {}


SHOPS_BY_TOKEN = load_shops_by_token()


# ============================================================
# Optional Google Calendar support (safe no-op if not set)
# ============================================================

def get_calendar_service():
    """
    Uses GOOGLE_SERVICE_ACCOUNT_JSON env var (full JSON for a service account).
    If anything is missing / broken, returns None and we just skip calendar.
    """
    try:
        if not (service_account and build):
            return None

        sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not sa_json:
            return None

        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=creds)
        return service
    except Exception as e:
        logging.error(f"Calendar init failed: {e}")
        return None


def create_calendar_event_if_possible(shop, customer_info, summary_text):
    """
    Best-effort: if calendar is configured and shop has calendar_id, create event.
    NEVER raises – errors are logged only.
    """
    try:
        service = get_calendar_service()
        if not service:
            return

        calendar_id = shop.get("calendar_id")
        if not calendar_id:
            return

        # customer_info is free-form text; we just drop it into description.
        event_body = {
            "summary": f"AI estimate lead - {customer_info.get('name','Unknown')}",
            "description": summary_text,
            "start": {
                # If no specific time, just put tomorrow at 10:00
                "dateTime": (
                    customer_info.get("iso_datetime")
                    or (datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
                )
            },
            "end": {
                "dateTime": (
                    customer_info.get("iso_datetime")
                    or (datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
                )
            },
        }
        service.events().insert(calendarId=calendar_id, body=event_body).execute()
    except Exception as e:
        logging.error(f"Failed to create calendar event: {e}")


# ============================================================
# Image download & encoding
# ============================================================

async def download_twilio_image_as_data_url(url: str) -> str:
    """
    Download a Twilio media URL using SID+TOKEN auth, return a data: URL
    suitable for OpenAI vision (base64 image).
    """
    async with httpx.AsyncClient(timeout=20.0) as http:
        resp = await http.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
        b64 = base64.b64encode(resp.content).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64}"
        return data_url


# ============================================================
# AI damage analysis
# ============================================================

async def analyze_damage_with_ai(shop, media_data_urls):
    """
    Calls OpenAI vision model with one or more images and returns:
    (estimate_text_for_customer, raw_json_or_text)
    """
    shop_name = shop.get("name", "the shop")

    system_prompt = (
        f"You are an expert auto body damage estimator working for {shop_name} "
        "in Ontario, Canada. You see photos of collision damage.\n"
        "Your job:\n"
        "1) Determine overall damage severity: one of ['minor','moderate','severe'].\n"
        "2) Estimate a realistic repair cost range in CAD (cost_min and cost_max).\n"
        "3) List damaged areas/panels in plain language.\n"
        "4) Write a short, friendly explanation for the customer.\n"
        "Respond ONLY in valid JSON with this schema:\n"
        "{\n"
        '  "severity": "minor|moderate|severe",\n'
        '  "cost_min": 1200,\n'
        '  "cost_max": 2500,\n'
        '  "panels": ["rear bumper", "trunk lid"],\n'
        '  "summary": "2–4 sentence explanation for the customer."\n"
        "}\n"
        "Do not include any extra keys, comments, or text outside the JSON."
    )

    user_text = (
        "Analyze the vehicle damage from these photos and produce the JSON described. "
        "Assume typical Ontario collision repair shop rates in 2025."
    )

    content = [{"type": "text", "text": user_text}]
    for data_url in media_data_urls:
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        ai_message = completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        raise

    # Try to parse JSON strictly
    parsed = None
    try:
        parsed = json.loads(ai_message)
    except Exception:
        # Sometimes model might wrap in ```json```; try to extract
        try:
            import re

            match = re.search(r"\{.*\}", ai_message, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
        except Exception:
            parsed = None

    if not parsed or not isinstance(parsed, dict):
        # Fallback: just send raw text
        estimate_text = (
            f"AI Analysis (raw text):\n\n{ai_message}\n\n"
            "This is a preliminary visual estimate only."
        )
        return estimate_text, ai_message

    severity = parsed.get("severity", "unknown").capitalize()
    cost_min = parsed.get("cost_min")
    cost_max = parsed.get("cost_max")
    panels = parsed.get("panels") or []
    summary = parsed.get("summary") or ""

    cost_line = ""
    try:
        if cost_min is not None and cost_max is not None:
            cost_line = f"Estimated Cost (CAD): ${float(cost_min):,.0f} – ${float(cost_max):,.0f}\n"
    except Exception:
        cost_line = ""

    panels_line = ""
    if panels:
        panels_line = "Areas affected: " + ", ".join(panels) + "\n"

    estimate_text = (
        f"AI Damage Estimate for {shop_name}\n\n"
        f"Severity: {severity}\n"
        f"{cost_line}"
        f"{panels_line}\n"
        f"{summary}\n\n"
        "Note: This is a visual, preliminary estimate only. "
        "Final pricing may change after in-person inspection."
    )

    return estimate_text, json.dumps(parsed)


# ============================================================
# Twilio webhook
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request, token: str):
    # Validate shop token
    shop = SHOPS_BY_TOKEN.get(token)
    if not shop:
        logging.warning(f"Unknown webhook token: {token}")
        raise HTTPException(status_code=404, detail="Unknown shop")

    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia") or 0)

    logging.info(
        f"Incoming SMS from {from_number} for shop {shop.get('id')} "
        f"with {num_media} media items."
    )

    resp = MessagingResponse()

    # If no image, send welcome / instructions
    if num_media == 0:
        welcome = (
            f"👋 Welcome to {shop.get('name', 'our collision centre')}!\n\n"
            "Please send 1–3 clear photos of your vehicle damage (different angles "
            "if possible), and our AI estimator will give you a free preliminary estimate."
        )
        resp.message(welcome)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # We have at least one image
    try:
        # Download and encode images as data URLs
        media_urls = []
        data_urls = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            if not url:
                continue
            media_urls.append(url)
            data_url = await download_twilio_image_as_data_url(url)
            data_urls.append(data_url)

        if not data_urls:
            raise RuntimeError("No valid media URLs found.")

        # Run AI analysis
        estimate_text, raw_analysis = await analyze_damage_with_ai(shop, data_urls)

        # Save to DB (best effort; don't break SMS if DB fails)
        try:
            estimate_id = str(uuid.uuid4())
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO estimates (id, phone, shop_id, image_url, ai_analysis)
                        VALUES (:id, :phone, :shop_id, :image_url, :ai_analysis)
                        """
                    ),
                    {
                        "id": estimate_id,
                        "phone": from_number,
                        "shop_id": shop.get("id"),
                        "image_url": ",".join(media_urls),
                        "ai_analysis": raw_analysis,
                    },
                )
        except Exception as e:
            logging.error(f"Failed to save estimate to DB: {e}")

        # First message: confirmation
        confirm = (
            "📸 Thanks! We received your photos.\n\n"
            f"Our AI estimator for {shop.get('name','the shop')} "
            "is analyzing the damage now — here’s your preliminary estimate:"
        )
        resp.message(confirm)

        # Second message: estimate details + booking instructions
        booking_instructions = (
            "\n\nIf you’d like to book an appointment, reply with your:\n"
            "1) Full name\n"
            "2) Phone number\n"
            "3) Email\n"
            "4) Preferred date & time\n\n"
            "The shop team will confirm your booking and may follow up with questions."
        )

        resp.message(estimate_text + booking_instructions)

        return PlainTextResponse(str(resp), media_type="application/xml")

    except Exception as e:
        logging.error(f"AI processing error: {e}")
        error_msg = (
            "⚠️ AI Processing Error: I couldn't analyze your photos this time. "
            "Please try again in a few minutes, or contact the shop directly."
        )
        resp.message(error_msg)
        return PlainTextResponse(str(resp), media_type="application/xml")


# Simple healthcheck
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Estimator backend running."}
