# =========================
# main.py — RULES > AI
# =========================

import os
import uuid
import json
import base64
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------
# SHOP CONFIG
# -------------------------
SHOP_CONFIGS = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110
    }
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}

# -------------------------
# OPENAI
# -------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# -------------------------
# AI PROMPT
# -------------------------
SYSTEM_PROMPT = """You are an automotive collision damage vision system used by professional body shops.
Identify only visible damage. Use DRIVER POV. Never estimate hours or cost.
Return ONLY JSON matching the schema."""
USER_PROMPT = "Analyze vehicle damage photos and return structured JSON."

# -------------------------
# RULE DATA
# -------------------------
PART_POINTS = {
    "bumper": 2,
    "fender": 3,
    "headlight": 2,
    "taillight": 2,
    "hood": 3,
    "quarter_panel": 4,
    "wheel": 5,
    "tire": 3,
    "suspension": 7,
    "radiator_support": 6,
    "mirror": 1,
    "grille": 1
}

FORCE_POINTS = {
    "wheel_displacement": 9,
    "ride_height_asymmetry": 7,
    "airbag_deployed": 10,
    "curb_or_object_strike": 2,
    "debris_field_visible": 2
}

OP_HOURS = {
    "Replace bumper": (4, 7),
    "Refinish bumper": (2, 4),
    "Replace fender": (5, 9),
    "Refinish fender": (2, 4),
    "Replace headlight": (1, 3),
    "Repair hood": (4, 8),
    "Replace hood": (6, 10),
    "Inspect suspension": (2, 4),
    "Wheel alignment": (1, 2)
}

# -------------------------
# HELPERS
# -------------------------
def money(n: int) -> str:
    return f"${n:,}"

def severity_from_score(score: int) -> str:
    if score >= 22:
        return "Severe"
    if score >= 10:
        return "Moderate"
    return "Minor"

# -------------------------
# AI CALL
# -------------------------
async def analyze_photos(photos: List[UploadFile]) -> Optional[Dict[str, Any]]:
    blocks = [{"type": "text", "text": USER_PROMPT}]
    for p in photos:
        raw = await p.read()
        blocks.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}
        })

    try:
        r = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": blocks}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return None

# -------------------------
# ROUTES (UI UNCHANGED)
# -------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/quote?shop_id=miss")

@app.get("/quote", response_class=HTMLResponse)
def quote(shop_id: str = "miss"):
    name = SHOP_CONFIGS["miss"]["name"]
    return HTMLResponse(open("templates/landing.html").read().replace("{{SHOP}}", name))

@app.post("/estimate/api")
async def estimate_api(
    photos: List[UploadFile] = File(...),
    shop_key: str = Form("miss")
):
    ai = await analyze_photos(photos[:3])
    score = 0
    ops = []

    if ai:
        for p in ai.get("damaged_parts", []):
            score += PART_POINTS.get(p["part"], 0)
            if p["part"] == "bumper":
                ops += ["Replace bumper", "Refinish bumper"]
            if p["part"] == "fender":
                ops += ["Replace fender", "Refinish fender"]
            if p["part"] == "headlight":
                ops += ["Replace headlight"]
            if p["part"] == "hood":
                ops += ["Repair hood"]

        for k, v in ai.get("force_indicators", {}).items():
            if v:
                score += FORCE_POINTS.get(k, 0)
                if k in ("wheel_displacement", "ride_height_asymmetry"):
                    ops += ["Inspect suspension", "Wheel alignment"]

    severity = severity_from_score(score)

    hmin = sum(OP_HOURS[o][0] for o in set(ops))
    hmax = sum(OP_HOURS[o][1] for o in set(ops))

    if severity == "Severe":
        hmin = max(hmin, 22)
        hmax = max(hmax, 38)
    elif severity == "Moderate":
        hmin = max(hmin, 8)

    rate = SHOP_CONFIGS[shop_key]["labor_rate"]
    estimate_id = str(uuid.uuid4())

    ESTIMATES[estimate_id] = {
        "severity": severity,
        "confidence": "High" if ai else "Medium",
        "summary": "Visible collision damage detected. Further inspection recommended.",
        "damaged_areas": [p["part"].title() for p in ai.get("damaged_parts", [])] if ai else [],
        "operations": list(set(ops)),
        "labour_hours_min": hmin,
        "labour_hours_max": hmax,
        "cost_min": money(hmin * rate),
        "cost_max": money(hmax * rate),
        "risk_note": "Hidden damage is common in collision repairs."
    }

    return JSONResponse({"estimate_id": estimate_id})

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    d = ESTIMATES.get(id)
    if not d:
        return RedirectResponse("/")
    html = open("templates/result.html").read()
    for k, v in d.items():
        html = html.replace(f"{{{{{k}}}}}", str(v))
    return HTMLResponse(html)

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/quote?shop_id=miss")

