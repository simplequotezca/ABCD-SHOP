import uuid
from typing import Dict, Any, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------
# Shop configuration
# --------------------------------------------------------------------
SHOP_CONFIGS = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110,
    }
}

SHOP_ALIASES = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def resolve_shop(shop_key: Optional[str]):
    key = (shop_key or "miss").lower()
    return key, SHOP_CONFIGS.get(key, SHOP_CONFIGS["miss"])

def money(n: int) -> str:
    return f"${n:,}"

# --------------------------------------------------------------------
# AI PLACEHOLDER OUTPUT (to be replaced with vision model later)
# --------------------------------------------------------------------
def fake_ai_vision(_: bytes) -> Dict[str, Any]:
    return {
        "damage": [
            {"area": "front bumper", "side": "driver", "severity": "severe"},
            {"area": "fender", "side": "driver", "severity": "severe"},
            {"area": "headlight", "side": "driver", "severity": "moderate"},
        ],
        "operations": [
            {"type": "replace", "area": "front bumper"},
            {"type": "replace", "area": "fender"},
            {"type": "replace", "area": "headlight"},
        ],
        "structural_risk": True,
        "confidence": "High",
    }

# --------------------------------------------------------------------
# RULE ENGINE (THIS IS THE MONEY)
# --------------------------------------------------------------------
def apply_rules(ai: Dict[str, Any]) -> Dict[str, Any]:
    severity = "Minor"

    areas = {d["area"] for d in ai["damage"]}
    operations = {o["type"] for o in ai["operations"]}

    if ai.get("structural_risk"):
        severity = "Severe"
    elif "replace" in operations:
        severity = "Moderate"
    elif len(areas) >= 2:
        severity = "Moderate"

    if "suspension" in areas or "frame" in areas:
        severity = "Severe"

    hours = {
        "Minor": (3, 6),
        "Moderate": (8, 14),
        "Severe": (22, 38),
    }

    summaries = {
        "Minor": "Light cosmetic damage detected.",
        "Moderate": "Visible body damage detected. Further inspection recommended.",
        "Severe": "Significant damage with possible structural involvement.",
    }

    risks = {
        "Minor": "Final cost may vary after inspection.",
        "Moderate": "Hidden damage is common in moderate impacts.",
        "Severe": "Hidden damage is common in front-corner impacts.",
    }

    return {
        "severity": severity,
        "confidence": ai.get("confidence", "Medium"),
        "summary": summaries[severity],
        "damaged_areas": [
            f"{d['side'].capitalize()} {d['area']}" for d in ai["damage"]
        ],
        "operations": [
            f"{o['type'].capitalize()} {o['area']}" for o in ai["operations"]
        ],
        "hours_min": hours[severity][0],
        "hours_max": hours[severity][1],
        "risk_note": risks[severity],
    }

# --------------------------------------------------------------------
# HTML RENDERERS (UNCHANGED)
# --------------------------------------------------------------------
def render_landing(shop_key, shop_name):
    return f"""
<html>
<head>
<link rel="stylesheet" href="/static/style.css">
</head>
<body class="page">
<div class="card">
<h1>{shop_name}</h1>
<p>Upload photos to get a fast AI repair estimate.</p>
<a class="cta" href="/estimate?shop_key={shop_key}">Start Estimate</a>
</div>
</body>
</html>
"""

def render_upload(shop_key, shop_name):
    return f"""
<html>
<head><link rel="stylesheet" href="/static/style.css"></head>
<body class="page">
<div class="card">
<h2>{shop_name}</h2>
<form method="post" action="/estimate/api" enctype="multipart/form-data">
<input type="hidden" name="shop_key" value="{shop_key}">
<input type="file" name="photo" accept="image/*" required>
<button class="cta">Analyze</button>
</form>
<a class="backlink" href="/quote/{shop_key}">← Back</a>
</div>
</body>
</html>
"""

def render_result(data):
    return f"""
<html>
<head><link rel="stylesheet" href="/static/style.css"></head>
<body class="page">
<div class="card">
<h1>AI Estimate</h1>
<div class="pill">{data['severity']} • {data['confidence']} confidence</div>
<p>{data['summary']}</p>
<ul>{"".join(f"<li>{d}</li>" for d in data['damaged_areas'])}</ul>
<ul>{"".join(f"<li>{o}</li>" for o in data['operations'])}</ul>
<p>Labour: {data['hours_min']} – {data['hours_max']} hours</p>
<div class="big">{data['cost_min']} – {data['cost_max']}</div>
<div class="warning">{data['risk_note']}</div>
<a class="backlink" href="/quote/{data['shop_key']}">← Start over</a>
</div>
</body>
</html>
"""

# --------------------------------------------------------------------
# ROUTES (UNCHANGED)
# --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return render_landing("miss", SHOP_CONFIGS["miss"]["name"])

@app.get("/quote/{shop_key}", response_class=HTMLResponse)
def quote(shop_key: str):
    k, cfg = resolve_shop(shop_key)
    return render_landing(k, cfg["name"])

@app.get("/estimate", response_class=HTMLResponse)
def estimate(shop_key: str = "miss"):
    k, cfg = resolve_shop(shop_key)
    return render_upload(k, cfg["name"])

@app.post("/estimate/api")
async def estimate_api(photo: UploadFile = File(...), shop_key: str = Form("miss")):
    k, cfg = resolve_shop(shop_key)
    content = await photo.read()

    ai_raw = fake_ai_vision(content)
    result = apply_rules(ai_raw)

    rate = cfg["labor_rate"]
    estimate_id = str(uuid.uuid4())

    ESTIMATES[estimate_id] = {
        "shop_key": k,
        **result,
        "cost_min": money(result["hours_min"] * rate),
        "cost_max": money(result["hours_max"] * rate),
    }

    return RedirectResponse(f"/estimate/result?id={estimate_id}", status_code=303)

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    return render_result(ESTIMATES[id])
