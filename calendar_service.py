import os
import json
from typing import Dict, Any, List
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# LOAD SHOPS FROM ENV (LIST FORMAT)
# ============================================================
def load_shops() -> List[Dict[str, Any]]:
    raw = os.environ.get("SHOPS_JSON")
    if not raw:
        raise RuntimeError("SHOPS_JSON environment variable not set")

    try:
        shops = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid SHOPS_JSON JSON: {e}")

    if not isinstance(shops, list):
        raise RuntimeError("SHOPS_JSON must be a LIST of shop objects")

    return shops


def get_shop(shop_key: str) -> Dict[str, Any]:
    shops = load_shops()
    for shop in shops:
        if shop.get("shop_key") == shop_key:
            return shop
    raise RuntimeError(f"Shop '{shop_key}' not found in SHOPS_JSON")


# ============================================================
# GOOGLE CALENDAR CLIENT
# ============================================================
def get_calendar_service():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

    try:
        info = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {e}")

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

    return build("calendar", "v3", credentials=creds)


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
        raise RuntimeError(f"No calendar_id configured for shop '{shop_key}'")

    service = get_calendar_service()

    description_lines = [
        "🚗 New AI Estimate Booking",
        "",
        f"Customer: {customer.get('name')}",
        f"Phone: {customer.get('phone')}",
        f"Email: {customer.get('email')}",
        "",
        "AI Estimate Summary:",
        f"- Severity: {ai_summary.get('severity')}",
        f"- Confidence: {ai_summary.get('confidence')}",
        f"- Labour: {ai_summary.get('labor_hours_range')}",
        f"- Price: {ai_summary.get('price_range')}",
    ]

    damaged = ai_summary.get("damaged_parts") or []
    if damaged:
        description_lines.append("")
        description_lines.append("Damaged Areas:")
        for d in damaged:
            description_lines.append(f"- {d}")

    if photo_urls:
        description_lines.append("")
        description_lines.append("Photos:")
        for url in photo_urls:
            description_lines.append(url)

    event = {
        "summary": summary,
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": start_iso,
            "timeZone": "America/Toronto",
        },
        "end": {
            "dateTime": end_iso,
            "timeZone": "America/Toronto",
        },
    }

    created = (
        service.events()
        .insert(calendarId=calendar_id, body=event)
        .execute()
    )

    return created
