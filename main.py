import os
import uuid
import json
import base64
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from openai import OpenAI
from calendar_service import create_calendar_event

# ============================================================
# APP
# ============================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# ENV / CONFIG
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
AI_LABOR_INFLUENCE = float(os.getenv("AI_LABOR_INFLUENCE", "0.30"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ============================================================
# SHOP CONFIG
# ============================================================
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {"name": "Mississauga Collision Center", "labor_rate": 110}
}

SHOP_ALIASES = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
    "mississauga_collision_center": "miss",
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}

# ============================================================
# HELPERS
# ============================================================
def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    key = SHOP_ALIASES.get((shop_key or "miss").strip(), "miss")
    return key, SHOP_CONFIGS[key]

def money_fmt(n: int) -> str:
    return f"${n:,}"

# ============================================================
# AI JSON CONTRACT
# ============================================================
AI_VISION_JSON_SCHEMA = {
    "name": "collision_estimate",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
            "severity": {"type": "string", "enum": ["Minor", "Moderate", "Severe"]},
            "driver_pov": {"type": "boolean"},
            "impact_side": {"type": "string", "enum": ["Driver", "Passenger", "Front", "Rear", "Unknown"]},
            "damaged_areas": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "operations": {"type": "array", "items": {"type": "string"}, "maxItems": 18},
            "structural_possible": {"type": "boolean"},
            "mechanical_possible": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "confidence",
            "severity",
            "driver_pov",
            "impact_side",
            "damaged_areas",
            "operations",
            "structural_possible",
            "mechanical_possible",
            "notes",
        ],
    },
}

# ============================================================
# RULE OVERRIDES
# ============================================================
def apply_rule_overrides(ai: Dict[str, Any]) -> Dict[str, Any]:
    ai = dict(ai or {})
    ai["driver_pov"] = True

    if ai.get("confidence") not in ("Low", "Medium", "High"):
        ai["confidence"] = "Medium"
    if ai.get("severity") not in ("Minor", "Moderate", "Severe"):
        ai["severity"] = "Moderate"

    if ai.get("structural_possible") and ai["severity"] == "Minor":
        ai["severity"] = "Moderate"

    ops = list(dict.fromkeys(ai.get("operations", [])))
    for mandatory in [
        "Pre-scan (diagnostics)",
        "Post-scan (diagnostics)",
        "Measure/inspect for hidden damage",
    ]:
        if mandatory not in ops:
            ops.append(mandatory)

    ai["operations"] = ops[:18]
    ai["damaged_areas"] = ai.get("damaged_areas", [])[:12]
    return ai

# ============================================================
# LABOR LOGIC
# ============================================================
OP_HOURS = {
    "Replace bumper": (3, 6),
    "Repair bumper": (2, 4),
    "Replace fender": (3, 6),
    "Repair fender": (2, 5),
    "Replace headlight": (1, 2),
    "Pre-scan (diagnostics)": (0, 1),
    "Post-scan (diagnostics)": (0, 1),
    "Measure/inspect for hidden damage": (1, 2),
}

def rules_labor_range(ops: List[str], severity: str) -> Tuple[int, int]:
    mn = sum(OP_HOURS.get(o, (0, 0))[0] for o in ops)
    mx = sum(OP_HOURS.get(o, (0, 0))[1] for o in ops)

    if severity == "Minor":
        return max(mn, 3), max(mx, 6)
    if severity == "Moderate":
        return max(mn, 8), max(mx, 14)
    return max(mn, 18), max(mx, 33)

def blend_labor_ranges(rules_mn, rules_mx, ai_mn, ai_mx):
    w = min(0.85, max(0.0, AI_LABOR_INFLUENCE))
    mn = int(round((1 - w) * rules_mn + w * ai_mn))
    mx = int(round((1 - w) * rules_mx + w * ai_mx))
    return mn, max(mx, mn + 1)

# ============================================================
# AI VISION (PATCHED — HARDENED)
# ============================================================
async def ai_vision_analyze(files: List[UploadFile]) -> Dict[str, Any]:
    fallback = {
        "confidence": "Medium",
        "severity": "Moderate",
        "driver_pov": True,
        "impact_side": "Unknown",
        "damaged_areas": ["Bumper", "Fender", "Headlight"],
        "operations": ["Replace bumper", "Repair fender", "Replace headlight"],
        "structural_possible": False,
        "mechanical_possible": False,
        "notes": "AI unavailable. Rules-based estimate used.",
    }

    if not client:
        return apply_rule_overrides(fallback)

    image_parts = []
    for f in files:
        data = await f.read()
        if not data or len(data) > 6_000_000:
            continue
        b64 = base64.b64encode(data).decode("utf-8")
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{f.content_type};base64,{b64}"},
        })

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": [{"type": "text", "text": "Analyze collision damage conservatively."}] + image_parts},
            ],
            response_format={"type": "json_schema", "json_schema": AI_VISION_JSON_SCHEMA},
            temperature=0.4,
        )
        ai = json.loads(resp.choices[0].message.content or "{}")
        return apply_rule_overrides(ai)
    except Exception:
        return apply_rule_overrides(fallback)

# ============================================================
# ROUTES (UI + API + BOOKING)
# ============================================================
@app.get("/")
def root():
    return RedirectResponse("/quote?shop_id=miss")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/quote")
def quote(shop_id: str = "miss"):
    k, cfg = resolve_shop(shop_id)
    return HTMLResponse(f"<h1>{cfg['name']}</h1><a href='/estimate?shop_key={k}'>Start</a>")

@app.get("/estimate")
def upload_page(shop_key: str = "miss"):
    return HTMLResponse("""
        <form action="/estimate/api" method="post" enctype="multipart/form-data">
            <input type="file" name="photos" multiple required />
            <input type="hidden" name="shop_key" value="%s"/>
            <button>Analyze</button>
        </form>
    """ % shop_key)

@app.post("/estimate/api")
async def estimate_api(
    request: Request,
    photos: List[UploadFile] = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)
    ai = await ai_vision_analyze(photos[:3])

    rules_min, rules_max = rules_labor_range(ai["operations"], ai["severity"])
    hours_min, hours_max = blend_labor_ranges(rules_min, rules_max, rules_min, rules_max)

    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": ai["severity"],
        "confidence": ai["confidence"],
        "operations": ai["operations"],
        "damaged_areas": ai["damaged_areas"],
        "labour_hours_min": hours_min,
        "labour_hours_max": hours_max,
        "cost_min": money_fmt(hours_min * cfg["labor_rate"]),
        "cost_max": money_fmt(hours_max * cfg["labor_rate"]),
        "estimate_id": estimate_id,
    }

    return {"estimate_id": estimate_id}

@app.get("/estimate/result")
def estimate_result(id: str):
    return ESTIMATES.get(id, {})

@app.post("/book")
def book(
    estimate_id: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
):
    est = ESTIMATES.get(estimate_id)
    if not est:
        return {"error": "Estimate not found"}

    start = datetime.fromisoformat(f"{date}T{time}")
    create_calendar_event(
        shop_key=est["shop_key"],
        start_iso=start.isoformat(),
        end_iso=(start + timedelta(hours=1)).isoformat(),
        summary="New Booking",
        customer={"name": name, "phone": phone, "email": email},
        ai_summary=est,
        photo_urls=[],
    )
    return {"status": "booked"}
