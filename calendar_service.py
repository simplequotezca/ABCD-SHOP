import os
import json
from typing import Dict, Any, List
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# LOAD SHOPS CONFIG FROM ENV
# ============================================================

SHOPS_JSON_RAW = os.getenv("SHOPS_JSON")

if not SHOPS_JSON_RAW:
    raise RuntimeError("SHOPS_JSON env var not set")

try:
    SHOPS = json.loads(SHOPS_JSON_RAW)
except Exception as e:
    raise RuntimeError(f"Failed to parse SHOPS_JSON: {e}")

if not isinstance(SHOPS, list):
    raise RuntimeError("SHOPS_JSON must be a list of shops")


def get_shop(shop_key: str) -> Dict[str, Any]:
    for shop in SHOPS:
        if shop.get("shop_key") == shop_key:
            return shop
    raise ValueError(f"Shop not found: {shop_key}")


# ============================================================
# GOOGLE CALENDAR CLIENT
# ============================================================

def get_calendar_service():
    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not credentials_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var not set")

    creds_info = json.loads(credentials_json)

    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

    return build("calendar", "v3", credentials=credentials)


# ============================================================
# CREATE CALENDAR EVENT
# ============================================================

def create_calendar_event(
    shop_key: str,
    start_iso: str,
    end_iso: str,
    summary: str,
    customer: Dict[str, str],
    photo_urls: List[str],
    ai_summary: Dict[str, Any],
) -> Dict[str, Any]:

    shop = get_shop(shop_key)
    calendar_id = shop.get("calendar_id")

    if not calendar_id:
        raise RuntimeError(f"No calendar_id configured for shop {shop_key}")

    service = get_calendar_service()

    description_lines = [
        f"Customer: {customer.get('name')}",
        f"Phone: {customer.get('phone')}",
        f"Email: {customer.get('email')}",
        "",
        "AI Estimate Summary:",
        f"Severity: {ai_summary.get('severity')}",
        f"Confidence: {ai_summary.get('confidence')}",
        f"Labour: {ai_summary.get('labor_hours_range')}",
        f"Price Range: {ai_summary.get('price_range')}",
        "",
        "Damaged Areas:",
    ]

    for part in ai_summary.get("damaged_parts", []):
        description_lines.append(f"- {part}")

    if photo_urls:
        description_lines.append("")
        description_lines.append("Uploaded Photos:")
        for url in photo_urls:
            description_lines.append(url)

    event_body = {
        "summary": summary,
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": start_iso,
        },
        "end": {
            "dateTime": end_iso,
        },
    }

    event = (
        service.events()
        .insert(calendarId=calendar_id, body=event_body)
        .execute()
    )

    # 🔑 IMPORTANT: RETURN A DICT (NOT A LIST)
    return {
        "eventId": event.get("id"),
        "htmlLink": event.get("htmlLink"),
    }
