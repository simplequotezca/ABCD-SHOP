# calendar.py
import os
import json
from typing import Any, Dict, List, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarError(Exception):
    pass


def _load_service_account_info() -> Dict[str, Any]:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise CalendarError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var.")

    try:
        info = json.loads(raw)
    except Exception as e:
        raise CalendarError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")

    # sanity
    for k in ["client_email", "private_key", "project_id"]:
        if k not in info or not info[k]:
            raise CalendarError(f"Service account JSON missing '{k}'.")

    return info


def get_calendar_service():
    info = _load_service_account_info()
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    # cache_discovery=False avoids occasional issues in serverless-ish envs
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def load_shops() -> Dict[str, Any]:
    """
    Expected SHOPS_JSON (example):
    {
      "mississauga-collision-center": {
        "display_name": "Mississauga Collision Center",
        "calendar_id": "yourcalendarid@group.calendar.google.com",
        "timezone": "America/Toronto"
      }
    }
    """
    raw = os.getenv("SHOPS_JSON", "").strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception as e:
        raise CalendarError(f"SHOPS_JSON is not valid JSON: {e}")


def get_shop_calendar_id(shop_key: str, shops: Optional[Dict[str, Any]] = None) -> str:
    shops = shops or load_shops()
    cfg = shops.get(shop_key) or {}
    cal_id = cfg.get("calendar_id") or ""
    if not cal_id:
        raise CalendarError(f"Missing calendar_id for shop '{shop_key}' in SHOPS_JSON.")
    return cal_id


def get_shop_timezone(shop_key: str, shops: Optional[Dict[str, Any]] = None) -> str:
    shops = shops or load_shops()
    cfg = shops.get(shop_key) or {}
    return cfg.get("timezone") or os.getenv("DEFAULT_TIMEZONE", "America/Toronto")


def build_booking_description(
    shop_key: str,
    customer: Dict[str, str],
    photo_urls: Optional[List[str]] = None,
    ai_summary: Optional[Dict[str, Any]] = None,
) -> str:
    lines = []
    lines.append("SimpleQuotez Booking Request")
    lines.append("")
    lines.append("Customer")
    lines.append(f"- Name: {customer.get('name','').strip()}")
    lines.append(f"- Phone: {customer.get('phone','').strip()}")
    lines.append(f"- Email: {customer.get('email','').strip()}")
    if customer.get("vehicle"):
        lines.append(f"- Vehicle: {customer.get('vehicle','').strip()}")
    lines.append("")
    if ai_summary:
        # keep short; shops hate walls of text
        sev = ai_summary.get("severity")
        conf = ai_summary.get("confidence")
        labor = ai_summary.get("labor_hours_range")
        price = ai_summary.get("price_range")
        if sev or conf:
            lines.append(f"AI: {sev or '—'} ({conf or '—'})")
        if labor:
            lines.append(f"Labour (AI): {labor}")
        if price:
            lines.append(f"Estimate (AI): {price}")
        parts = ai_summary.get("damaged_parts") or []
        if parts:
            lines.append("Damage areas (AI):")
            for p in parts[:12]:
                lines.append(f"- {p}")
        lines.append("")

    if photo_urls:
        lines.append("Photos")
        for i, url in enumerate(photo_urls[:5], start=1):
            lines.append(f"- Photo {i}: {url}")
        lines.append("")

    lines.append("Note: Photo-based preliminary estimate only. Final pricing after in-person inspection/teardown.")
    return "\n".join(lines).strip()


def create_calendar_event(
    *,
    shop_key: str,
    start_iso: str,
    end_iso: str,
    summary: str,
    customer: Dict[str, str],
    location: Optional[str] = None,
    photo_urls: Optional[List[str]] = None,
    ai_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    start_iso / end_iso should be ISO8601 with timezone offset, or naive ISO used with timeZone field.
    Example:
      start_iso="2025-12-21T10:00:00"
      end_iso="2025-12-21T11:00:00"
    """
    shops = load_shops()
    calendar_id = get_shop_calendar_id(shop_key, shops)
    tz = get_shop_timezone(shop_key, shops)

    description = build_booking_description(
        shop_key=shop_key,
        customer=customer,
        photo_urls=photo_urls,
        ai_summary=ai_summary,
    )

    body = {
        "summary": summary,
        "location": location or "",
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": tz},
        "end": {"dateTime": end_iso, "timeZone": tz},
        "reminders": {"useDefault": True},
    }

    try:
        svc = get_calendar_service()
        event = svc.events().insert(calendarId=calendar_id, body=body).execute()
        return {
            "ok": True,
            "event_id": event.get("id"),
            "htmlLink": event.get("htmlLink"),
            "calendar_id": calendar_id,
        }
    except HttpError as e:
        raise CalendarError(f"Google Calendar API error: {e}")
    except Exception as e:
        raise CalendarError(f"Calendar create failed: {e}")


def quick_service_check() -> Tuple[bool, str]:
    """
    Minimal check: can we list calendars? (does NOT guarantee shop calendar permission,
    but confirms service account creds are valid + API works)
    """
    try:
        svc = get_calendar_service()
        svc.calendarList().list(maxResults=1).execute()
        return True, "Service account auth + Calendar API OK."
    except Exception as e:
        return False, f"Service account / Calendar API failed: {e}"
