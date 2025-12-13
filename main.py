from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="SimpleQuotez V4")

# =========================
# SHOP CONFIG (SINGLE SOURCE OF TRUTH)
# =========================

SHOPS = {
    "miss": {
        "slug": "mississauga-collision-center",
        "name": "Mississauga Collision Center",
        "calendar_id": "0eec1cd6a07f5e8565e63bf0b4f5dbaf8b42f0ce183afe241cbf5f1dfe097fed@group.calendar.google.com",
    }
}

SLUG_TO_SHOP = {v["slug"]: k for k, v in SHOPS.items()}

# =========================
# STATIC FILES (UI)
# =========================

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# HEALTH
# =========================

@app.get("/api/health")
def health():
    return {"status": "ok"}

# =========================
# CLEAN PER-SHOP URL
# =========================

@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    shop_id = SLUG_TO_SHOP.get(shop_slug)
    if not shop_id:
        raise HTTPException(status_code=404, detail="Shop not found")

    shop = SHOPS[shop_id]

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{shop['name']} – AI Estimate</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="card">
        <img src="/static/logo.png" class="logo" />
        <h1>{shop['name']}</h1>
        <p>Upload photos to get a fast AI repair estimate.</p>
        <button onclick="alert('Upload flow coming next')">Start Estimate</button>
        <div class="note">
            Preliminary estimate · Final pricing after inspection
        </div>
    </div>
</body>
</html>
"""

# =========================
# CALENDAR API (REAL ENDPOINT)
# =========================

class CalendarBooking(BaseModel):
    shop_id: str
    customer_name: str
    phone: str
    email: Optional[str] = None
    start_time: str
    end_time: str
    notes: Optional[str] = None

@app.post("/api/calendar/book")
def book_calendar(payload: CalendarBooking):
    shop = SHOPS.get(payload.shop_id)
    if not shop:
        raise HTTPException(status_code=404, detail="Shop not found")

    # REAL calendar write happens next phase
    # This endpoint is intentionally stable + ready

    return {
        "status": "success",
        "shop": shop["name"],
        "calendar_id": shop["calendar_id"],
        "start": payload.start_time,
        "end": payload.end_time,
    }

@app.get("/api/calendar/health/{shop_id}")
def calendar_health(shop_id: str):
    shop = SHOPS.get(shop_id)
    if not shop:
        raise HTTPException(status_code=404, detail="Shop not found")

    return {
        "status": "ok",
        "shop": shop["name"],
        "calendar_id": shop["calendar_id"]
    }
