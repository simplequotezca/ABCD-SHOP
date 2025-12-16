import asyncio
import uuid
import hashlib
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -----------------------------
# Static files
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Shops
# -----------------------------
SHOPS = {
    "miss": "Mississauga Collision Center",
    "mississauga-collision-center": "Mississauga Collision Center",
    "mississauga-collision-centre": "Mississauga Collision Centre",
}

def resolve_shop(key: Optional[str]) -> str:
    if not key:
        return "AI Estimate"
    k = key.lower().strip()
    return SHOPS.get(k, k.replace("-", " ").title())

# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ============================================================
# SEVERITY → PARTS → HOURS LOGIC (LOGIC ONLY)
# ============================================================

def seed_from_photo(photo_bytes: bytes) -> int:
    return int(hashlib.sha256(photo_bytes).hexdigest()[:8], 16)

def decide_severity(seed: int) -> str:
    if seed % 10 < 2:
        return "Minor"
    elif seed % 10 < 6:
        return "Moderate"
    return "Severe"

def confidence_for(severity: str) -> str:
    return {
        "Minor": "High",
        "Moderate": "Medium–High",
        "Severe": "Medium"
    }[severity]

def parts_and_ops(severity: str) -> (List[str], List[str]):
    if severity == "Minor":
        return (
            ["Bumper", "Fender liner"],
            ["Repair bumper", "Replace clips/liners"]
        )
    if severity == "Moderate":
        return (
            ["Bumper", "Fender", "Headlight"],
            ["Replace bumper", "Replace fender", "Replace headlight"]
        )
    return (
        ["Bumper", "Fender", "Headlight", "Hood", "Suspension"],
        [
            "Replace bumper",
            "Replace fender",
            "Replace headlight",
            "Repair hood",
            "Inspect suspension"
        ]
    )

def labour_hours(severity: str, seed: int) -> (int, int):
    if severity == "Minor":
        return 6 + seed % 3, 12 + seed % 3
    if severity == "Moderate":
        return 14 + seed % 4, 24 + seed % 4
    return 26 + seed % 6, 40 + seed % 6

def cost_from_hours(hours_min: int, hours_max: int, severity: str) -> Dict[str, int]:
    rate = 110
    labour_min = hours_min * rate
    labour_max = hours_max * rate

    if severity == "Minor":
        parts_min, parts_max = 400, 900
    elif severity == "Moderate":
        parts_min, parts_max = 1400, 3000
    else:
        parts_min, parts_max = 2400, 5200

    return {
        "cost_min": labour_min + parts_min,
        "cost_max": labour_max + parts_max
    }

def summary_for(severity: str) -> str:
    return {
        "Minor": "Light cosmetic damage likely limited to exterior bolt-on components.",
        "Moderate": "Moderate front-corner damage affecting exterior panels and lighting.",
        "Severe": "Significant front-left damage with possible structural or mechanical involvement."
    }[severity]

def risk_note_for(severity: str) -> str:
    return {
        "Minor": "Hidden fastener or bracket damage may affect final pricing.",
        "Moderate": "Hidden damage is common behind bumpers and headlights.",
        "Severe": "Hidden damage is common in high-impact corner collisions."
    }[severity]

# ============================================================
# STEP 1 — LANDING (UNCHANGED)
# ============================================================
@app.get("/quote", response_class=HTMLResponse)
def landing_query(shop_id: Optional[str] = None):
    shop_key = shop_id or "miss"
    shop_name = resolve_shop(shop_key)
    return HTMLResponse(render_landing(shop_name, shop_key))

@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def landing_slug(shop_slug: str):
    shop_name = resolve_shop(shop_slug)
    return HTMLResponse(render_landing(shop_name, shop_slug))

def render_landing(shop_name: str, shop_key: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{shop_name} – AI Estimate</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
<div class="page">
  <div class="card hero">
    <img src="/static/logo.png" class="logo" alt="{shop_name}" onerror="this.style.display='none'"/>
    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">Upload photos to get a fast AI repair estimate.</div>
    <a href="/estimate?shop_key={shop_key}" class="cta">Start Estimate</a>
    <div class="hint">
      <div class="hint-title">Best results with 3 photos:</div>
      <ul class="hint-list">
        <li>Overall damage</li>
        <li>Close-up</li>
        <li>Side angle</li>
      </ul>
    </div>
    <div class="fineprint">
      Photo-based preliminary range only. Final pricing is confirmed after teardown and in-person inspection.
    </div>
  </div>
</div>
</body>
</html>
"""

# ============================================================
# STEP 2 — UPLOAD (UNCHANGED)
# ============================================================
@app.get("/estimate", response_class=HTMLResponse)
def upload_page(shop_key: Optional[str] = None):
    key = shop_key or "miss"
    shop_name = resolve_shop(key)
    return HTMLResponse(render_upload(shop_name, key))

def render_upload(shop_name: str, shop_key: str) -> str:
    return f"""(UNCHANGED HTML FROM YOUR BASELINE)"""

# ============================================================
# STEP 3 — RESULT (UNCHANGED)
# ============================================================
ESTIMATES = {}

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return HTMLResponse("<h3>Estimate not found</h3>")
    def bullets(items: List[str]) -> str:
        return "".join(f"<li>{i}</li>" for i in items)
    return f"""(UNCHANGED HTML FROM YOUR BASELINE)"""

# ============================================================
# ESTIMATE API — LOGIC-ONLY REPLACEMENT
# ============================================================

@app.post("/api/estimate")
async def estimate_api(photo: UploadFile = File(...)):
    photo_bytes = await photo.read()

    # Deterministic seed so same photo = same estimate
    seed = _seed_from_photo(photo_bytes)

    # Core logic
    severity = decide_severity(seed)
    confidence = severity_confidence(severity)
    parts, operations = parts_and_operations(severity)
    hours_min, hours_max = labour_hours_range(severity, seed)
    costs = estimate_cost(severity, hours_min, hours_max)

    # Store result exactly how your UI already expects it
    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = {
        "severity": severity,
        "confidence": confidence,
        "summary": severity_summary(severity),
        "damaged_areas": parts,
        "operations": operations,
        "labour_hours": f"{hours_min}–{hours_max}",
        "labour_cost": f"${costs['labour_min']:,} – ${costs['labour_max']:,}",
        "parts_cost": f"${costs['parts_min']:,} – ${costs['parts_max']:,}",
        "cost_min": costs["total_min"],
        "cost_max": costs["total_max"],
        "risk_note": risk_note(severity),
    }

    return {"estimate_id": estimate_id}


# ============================================================
# LOGIC FOUNDATION — SEVERITY → PARTS → HOURS
# (SAFE TO APPEND AT BOTTOM OF FILE)
# ============================================================

import hashlib
from typing import List, Dict, Tuple


def _seed_from_photo(photo_bytes: bytes) -> int:
    """
    Deterministic seed so the same photo always yields the same result.
    """
    return int(hashlib.sha256(photo_bytes).hexdigest()[:8], 16)


def decide_severity(seed: int) -> str:
    """
    Severity decision (rule-based, deterministic).
    """
    if seed % 10 < 2:
        return "Minor"
    elif seed % 10 < 6:
        return "Moderate"
    return "Severe"


def severity_confidence(severity: str) -> str:
    return {
        "Minor": "High",
        "Moderate": "Medium–High",
        "Severe": "Medium",
    }[severity]


def parts_and_operations(severity: str) -> Tuple[List[str], List[str]]:
    if severity == "Minor":
        return (
            ["Bumper", "Fender liner"],
            ["Repair bumper", "Replace clips / liners"],
        )

    if severity == "Moderate":
        return (
            ["Bumper", "Fender", "Headlight"],
            [
                "Replace bumper",
                "Replace fender",
                "Replace headlight",
            ],
        )

    # Severe
    return (
        ["Bumper", "Fender", "Headlight", "Hood", "Suspension"],
        [
            "Replace bumper",
            "Replace fender",
            "Replace headlight",
            "Repair hood",
            "Inspect suspension",
        ],
    )


def labour_hours_range(severity: str, seed: int) -> Tuple[int, int]:
    if severity == "Minor":
        return 6 + seed % 3, 12 + seed % 3

    if severity == "Moderate":
        return 14 + seed % 4, 24 + seed % 4

    return 26 + seed % 6, 40 + seed % 6


def estimate_cost(
    severity: str,
    hours_min: int,
    hours_max: int,
    labour_rate: int = 110,
) -> Dict[str, int]:
    labour_min = hours_min * labour_rate
    labour_max = hours_max * labour_rate

    if severity == "Minor":
        parts_min, parts_max = 400, 900
    elif severity == "Moderate":
        parts_min, parts_max = 1400, 3000
    else:
        parts_min, parts_max = 2400, 5200

    return {
        "labour_min": labour_min,
        "labour_max": labour_max,
        "parts_min": parts_min,
        "parts_max": parts_max,
        "total_min": labour_min + parts_min,
        "total_max": labour_max + parts_max,
    }


def severity_summary(severity: str) -> str:
    return {
        "Minor": "Light cosmetic damage likely limited to exterior bolt-on components.",
        "Moderate": "Moderate front-corner damage affecting exterior panels and lighting.",
        "Severe": "Significant front-corner damage with possible structural or mechanical involvement.",
    }[severity]


def risk_note(severity: str) -> str:
    return {
        "Minor": "Minor hidden fastener or bracket damage may affect final pricing.",
        "Moderate": "Hidden damage is common behind bumpers and headlights.",
        "Severe": "Hidden damage is common in high-impact corner collisions.",
    }[severity]
