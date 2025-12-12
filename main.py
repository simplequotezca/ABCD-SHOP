import os
import json
import base64
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from starlette.concurrency import run_in_threadpool

from openai import OpenAI
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


# ============================================================
# FastAPI app + CORS
# ============================================================

app = FastAPI(title="SimpleQuotez – AI Damage Estimator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later per domain if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OpenAI client (GPT-5.2)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing")

client = OpenAI(api_key=OPENAI_API_KEY)

GPT_MODEL = "gpt-5.2"  # vision + JSON output


# ============================================================
# Shop configuration (multi-shop ready)
# ============================================================

SHOPS: List[Dict[str, Any]] = [
    {
        "id": "miss",
        "name": "Mississauga Collision Center",
        "timezone": "America/Toronto",
        "calendar_id": "0eec1cd6a07f5e8565e63bf0b4f5dbaf8b42f0ce183afe241cbf5f1dfe097fed@group.calendar.google.com",
        "pricing": {
            "labor_rates": {
                "body": 95.0,
                "paint": 105.0,
            },
            "materials_rate": 38.0,
            "base_floor": {
                "minor_min": 450.0,
                "minor_max": 1100.0,
                "moderate_min": 1500.0,
                "moderate_max": 4000.0,
                "severe_min": 4000.0,
                "severe_max": 9000.0,
            },
            # multipliers to push estimates toward realistic, not cheap
            "severity_multiplier": {
                "minor": 1.0,
                "moderate": 1.15,
                "severe": 1.35,
            },
        },
    }
    # Add more shops here later; just match this structure.
]


def get_shop(shop_id: str) -> Dict[str, Any]:
    for shop in SHOPS:
        if shop["id"] == shop_id:
            return shop
    raise HTTPException(status_code=404, detail=f"Unknown shop_id '{shop_id}'")


# ============================================================
# Google Calendar client
# ============================================================

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]

def _build_calendar_service():
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON environment variable is missing. "
            "Paste your service account JSON into this env var in Railway."
        )
    info = json.loads(sa_json)
    creds = Credentials.from_service_account_info(info, scopes=CALENDAR_SCOPES)
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return service


async def get_calendar_service():
    # run blocking client creation off the event loop
    return await run_in_threadpool(_build_calendar_service)


# ============================================================
# Pydantic models (flexible to avoid frontend breakage)
# ============================================================

class EstimateResult(BaseModel):
    severity: str
    estimate_min: float
    estimate_max: float
    currency: str = "CAD"
    areas: List[str] = Field(default_factory=list)
    summary: str
    tags: List[str] = Field(default_factory=list)


class BookingEstimate(BaseModel):
    severity: str
    estimate_min: float
    estimate_max: float
    currency: str = "CAD"
    areas: List[str] = Field(default_factory=list)
    summary: str


class BookingRequest(BaseModel):
    """
    Designed to be tolerant of whatever the frontend sends.
    Extra fields are allowed and just forwarded into the event description.
    """
    model_config = ConfigDict(extra="allow")

    shop_id: str
    start_iso: str
    end_iso: str
    customer_name: str
    customer_email: str
    customer_phone: str
    vehicle: Optional[str] = None
    estimate: Optional[BookingEstimate] = None
    photos: Optional[List[str]] = None  # URLs or filenames, whatever you send


class Slot(BaseModel):
    start_iso: str
    end_iso: str


class SlotsResponse(BaseModel):
    shop_id: str
    date: str
    timezone: str
    slots: List[Slot]


# ============================================================
# Utility helpers
# ============================================================

def encode_upload_to_data_url(upload: UploadFile) -> str:
    """Convert uploaded image to base64 data URL for OpenAI vision."""
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload")
    mime = upload.content_type or "image/jpeg"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def compute_price_from_ai(
    shop: Dict[str, Any],
    ai_payload: Dict[str, Any],
) -> EstimateResult:
    pricing = shop["pricing"]
    base_floor = pricing["base_floor"]
    severity_multiplier = pricing.get("severity_multiplier", {})
    labor_rates = pricing["labor_rates"]

    severity_raw = (ai_payload.get("severity") or "moderate").lower()
    if severity_raw not in ("minor", "moderate", "severe"):
        severity_raw = "moderate"

    body_hours = float(ai_payload.get("body_labor_hours") or 3.0)
    paint_hours = float(ai_payload.get("paint_labor_hours") or 2.0)

    body_rate = float(labor_rates["body"])
    paint_rate = float(labor_rates["paint"])

    labor_cost = body_hours * body_rate + paint_hours * paint_rate
    # simple materials model: hourly materials + 25% of labor
    materials_cost = (body_hours + paint_hours) * float(pricing["materials_rate"])
    materials_cost += labor_cost * 0.25

    if severity_raw == "minor":
        base_min = base_floor["minor_min"]
        base_max = base_floor["minor_max"]
    elif severity_raw == "severe":
        base_min = base_floor["severe_min"]
        base_max = base_floor["severe_max"]
    else:
        base_min = base_floor["moderate_min"]
        base_max = base_floor["moderate_max"]

    est_min = base_min + 0.9 * labor_cost + 0.8 * materials_cost
    est_max = base_max + 1.25 * labor_cost + 1.1 * materials_cost

    mult = float(severity_multiplier.get(severity_raw, 1.0))
    est_min *= mult
    est_max *= mult

    areas = ai_payload.get("damaged_areas") or []
    if isinstance(areas, str):
        areas = [a.strip() for a in areas.split(",") if a.strip()]

    summary = ai_payload.get("summary") or ai_payload.get("damage_summary") or ""
    summary = summary.strip()

    tags: List[str] = []
    if ai_payload.get("primary_damage_type"):
        tags.append(ai_payload["primary_damage_type"])
    tags.append("local repair rates")

    return EstimateResult(
        severity=severity_raw.capitalize(),
        estimate_min=round(est_min, -1),  # nearest 10
        estimate_max=round(est_max, -1),
        areas=areas,
        summary=summary,
        tags=tags,
    )


def build_event_description(req: BookingRequest) -> str:
    lines = [
        f"Booked via: SimpleQuotez AI Damage Estimator",
        "",
        f"Customer name: {req.customer_name}",
        f"Customer phone: {req.customer_phone}",
        f"Customer email: {req.customer_email}",
    ]
    if req.vehicle:
        lines.append(f"Vehicle: {req.vehicle}")

    if req.estimate:
        lines.extend(
            [
                "",
                "AI PRELIMINARY ESTIMATE (visual only):",
                f"  Severity: {req.estimate.severity}",
                f"  Range: {req.estimate.estimate_min:.0f} – {req.estimate.estimate_max:.0f} {req.estimate.currency}",
            ]
        )
        if req.estimate.areas:
            lines.append(f"  Areas: {', '.join(req.estimate.areas)}")
        if req.estimate.summary:
            lines.append("")
            lines.append(req.estimate.summary)

    if req.photos:
        lines.extend(["", "Photo links:"])
        for p in req.photos:
            lines.append(f"  - {p}")

    # dump any extra JSON keys so the shop still sees everything
    extra_keys = sorted(
        k for k in req.model_extra.keys() if k not in {
            "shop_id",
            "start_iso",
            "end_iso",
            "customer_name",
            "customer_email",
            "customer_phone",
            "vehicle",
            "estimate",
            "photos",
        }
    )
    if extra_keys:
        lines.append("")
        lines.append("Extra booking data:")
        for k in extra_keys:
            lines.append(f"  {k}: {req.model_extra[k]}")

    return "\n".join(lines)


# ============================================================
# OpenAI – vision estimate
# ============================================================

async def call_gpt_for_estimate(
    shop: Dict[str, Any],
    data_urls: List[str],
) -> Dict[str, Any]:
    if not data_urls:
        raise HTTPException(status_code=400, detail="No images provided")

    content_blocks: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are an expert auto body estimator in Ontario, Canada. "
                "Analyze these vehicle damage photos from the driver's point of view. "
                "Respond ONLY with a JSON object, no explanation text."
            ),
        }
    ]

    for url in data_urls:
        content_blocks.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise collision-repair estimator. "
                "You never under-estimate severe damage. "
                "Always think in terms of real-world body+paint hours and common replacement parts."
            ),
        },
        {
            "role": "user",
            "content": content_blocks,
        },
    ]

    # The model must return strict JSON
    response = await run_in_threadpool(
        lambda: client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=550,
        )
    )

    try:
        text = response.choices[0].message.content
        payload = json.loads(text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response as JSON: {e}",
        )

    # Expected JSON schema (we keep it flexible):
    # {
    #   "severity": "minor|moderate|severe",
    #   "damaged_areas": ["front bumper", "left fender", "headlight"],
    #   "body_labor_hours": 10.5,
    #   "paint_labor_hours": 7.0,
    #   "damage_summary": "..."
    # }
    return payload


# ============================================================
# API endpoints
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/shops")
def list_shops():
    """Frontend can call this to get names + config."""
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "timezone": s["timezone"],
        }
        for s in SHOPS
    ]


@app.post("/api/estimate")
async def create_estimate(
    shop_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Step 1 → 2: user uploads 1–3 images.
    Frontend sends multipart/form-data with:
      - shop_id
      - files[] (1–3 images)
    """
    shop = get_shop(shop_id)

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    data_urls: List[str] = []
    for upload in files:
        data_urls.append(encode_upload_to_data_url(upload))

    ai_payload = await call_gpt_for_estimate(shop, data_urls)
    estimate = compute_price_from_ai(shop, ai_payload)

    return {
        "shop_id": shop_id,
        "shop_name": shop["name"],
        "severity": estimate.severity,
        "estimate_min": estimate.estimate_min,
        "estimate_max": estimate.estimate_max,
        "currency": estimate.currency,
        "areas": estimate.areas,
        "summary": estimate.summary,
        "tags": estimate.tags,
        "raw_ai": ai_payload,  # useful for debugging but can be removed later
    }


@app.get("/api/slots", response_model=SlotsResponse)
async def get_available_slots(shop_id: str, date_str: str):
    """
    Step 4 – before choosing a time.
    Frontend calls /api/slots?shop_id=miss&date_str=2025-12-12
    We return open 1-hour slots between 9am–5pm.
    """
    shop = get_shop(shop_id)
    tz = ZoneInfo(shop.get("timezone", "America/Toronto"))

    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format; use YYYY-MM-DD")

    start_dt = datetime.combine(target_date, time(9, 0), tzinfo=tz)
    end_dt = datetime.combine(target_date, time(17, 0), tzinfo=tz)

    service = await get_calendar_service()

    body = {
        "timeMin": start_dt.isoformat(),
        "timeMax": end_dt.isoformat(),
        "timeZone": shop["timezone"],
        "items": [{"id": shop["calendar_id"]}],
    }

    fb = await run_in_threadpool(lambda: service.freebusy().query(body=body).execute())
    busy = fb["calendars"][shop["calendar_id"]]["busy"]

    busy_ranges = [
        (
            datetime.fromisoformat(b["start"]).astimezone(tz),
            datetime.fromisoformat(b["end"]).astimezone(tz),
        )
        for b in busy
    ]

    def is_free(slot_start: datetime, slot_end: datetime) -> bool:
        for b_start, b_end in busy_ranges:
            if slot_start < b_end and slot_end > b_start:
                return False
        return True

    slots: List[Slot] = []
    current = start_dt
    while current < end_dt:
        slot_end = current + timedelta(hours=1)
        if is_free(current, slot_end):
            slots.append(
                Slot(
                    start_iso=current.isoformat(),
                    end_iso=slot_end.isoformat(),
                )
            )
        current = slot_end

    return SlotsResponse(
        shop_id=shop_id,
        date=date_str,
        timezone=shop["timezone"],
        slots=slots,
    )


@app.post("/api/book")
async def book_appointment(req: BookingRequest):
    """
    Step 4 → booking confirmed.
    Frontend should POST JSON like:
    {
      "shop_id": "miss",
      "start_iso": "2025-12-12T10:00:00-05:00",
      "end_iso": "2025-12-12T11:00:00-05:00",
      "customer_name": "SJ",
      "customer_email": "shiran.jey@hotmail.com",
      "customer_phone": "6477026465",
      "vehicle": "2018 Toyota Camry",
      "estimate": { ... },
      "photos": ["https://link-to-photo-1", "..."]
    }
    Extra keys are accepted and will be dumped into the event description.
    """
    shop = get_shop(req.shop_id)
    tz = ZoneInfo(shop.get("timezone", "America/Toronto"))

    try:
        start = datetime.fromisoformat(req.start_iso)
        end = datetime.fromisoformat(req.end_iso)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start_iso or end_iso")

    if start.tzinfo is None:
        start = start.replace(tzinfo=tz)
    if end.tzinfo is None:
        end = end.replace(tzinfo=tz)

    service = await get_calendar_service()

    description = build_event_description(req)

    event_body = {
        "summary": f"{shop['name']} – AI estimate appointment",
        "start": {"dateTime": start.isoformat(), "timeZone": shop["timezone"]},
        "end": {"dateTime": end.isoformat(), "timeZone": shop["timezone"]},
        "description": description,
    }

    try:
        event = await run_in_threadpool(
            lambda: service.events()
            .insert(calendarId=shop["calendar_id"], body=event_body)
            .execute()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create calendar event: {e}",
        )

    return {"ok": True, "event_id": event.get("id")}
