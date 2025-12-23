import os
import json
from typing import Dict, Any, List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build


# ============================================================
# GOOGLE CALENDAR SETUP
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]

SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SHOPS_JSON_RAW = os.getenv("SHOPS_JSON")


def _load_service():
    if not SERVICE_ACCOUNT_JSON:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var not set")

    creds_info = json.loads(SERVICE_ACCOUNT_JSON)

    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES,
    )

    return build("calendar", "v3", credentials=credentials)


def _load_shops() -> List[Dict[str, Any]]:
    if not SHOPS_JSON_RAW:
        raise RuntimeError("SHOPS_JSON env var not set")

    data = json.loads(SHOPS_JSON_RAW)

    if not isinstance(data, list):
        raise RuntimeError("SHOPS_JSON must be a list")

    return data


def _get_shop(shop_key: str) -> Dict[str, Any]:
    shops = _load_shops()

    for shop in shops:
        if shop.get("shop_key") == shop_key:
            return shop

    raise RuntimeError(f"Shop not found for key: {shop_key}")


# ============================================================
# PUBLIC API — USED BY main.py
# ============================================================

def create_calendar_event(
    *,
    shop_key: str,
    start_iso: str,
    end_iso: str,
    summary: str,
    customer: Dict[str, str],
    photo_urls: List[str],
    ai_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a Google Calendar event with:
    - Shop email as attendee
    - Immediate email + popup notifications
    """

    service = _load_service()
    shop = _get_shop(shop_key)

    calendar_id = shop.get("calendar_id")
    if not calendar_id:
        raise RuntimeError("calendar_id missing for shop")

    shop_email = (
        shop.get("notification_email")
        or shop.get("email")
        or os.getenv("SHOP_NOTIFICATION_EMAIL")
    )

    # --------------------------------------------------------
    # Build event description
    # --------------------------------------------------------

    description_lines = [
        f"Customer name: {customer.get('name')}",
        f"Phone: {customer.get('phone')}",
        f"Email: {customer.get('email')}",
        "",
        "AI Estimate Summary:",
        f"Severity: {ai_summary.get('severity')}",
        f"Confidence: {ai_summary.get('confidence')}",
        f"Labor hours: {ai_summary.get('labor_hours_range')}",
        f"Price range: {ai_summary.get('price_range')}",
        "",
        "Damaged areas:",
    ]

    for part in ai_summary.get("damaged_parts", []):
        description_lines.append(f"- {part}")

    if photo_urls:
        description_lines.append("")
        description_lines.append("Photos:")
        for url in photo_urls:
            description_lines.append(url)

    # --------------------------------------------------------
    # Build calendar event
    # --------------------------------------------------------

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
        "attendees": (
            [{"email": shop_email}] if shop_email else []
        ),
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 0},
                {"method": "popup", "minutes": 0},
                {"method": "popup", "minutes": 15},
            ],
        },
    }

    created = (
        service.events()
        .insert(
            calendarId=calendar_id,
            body=event,
        )
        .execute()
    )

    return created
