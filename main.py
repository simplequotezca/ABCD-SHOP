import uuid
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -------------------------
# Static files
# -------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------
# Shop configuration
# -------------------------
SHOP_CONFIGS = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110
    }
}

SHOP_ALIASES = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss"
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Utilities
# -------------------------
def resolve_shop(key: Optional[str]) -> Dict[str, Any]:
    k = (key or "miss").strip()
    canonical = SHOP_ALIASES.get(k, k)
    return SHOP_CONFIGS.get(canonical, SHOP_CONFIGS["miss"])

def money(n: int) -> str:
    return f"${n:,}"

# -------------------------
# RULE LAYER (authoritative)
# -------------------------
def rule_engine(ai: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rules override AI optimism.
    AI can only add detail, never reduce severity or hours.
    """

    severity = ai.get("severity", "Moderate")
    areas = ai.get("damaged_areas", [])
    ops = ai.get("operations", [])

    if "Suspension" in areas or "Frame" in areas:
        severity = "Severe"

    OP_HOURS = {
        "Replace bumper": 4,
        "Repair bumper": 3,
        "Replace fender": 5,
        "Repair fender": 3,
        "Replace headlight": 2,
        "Repair hood": 4,
        "Inspect suspension": 3,
    }

    hours = sum(OP_HOURS.get(o, 2) for o in ops)

    if severity == "Minor":
        h_min, h_max = max(2, hours - 1), hours + 1
    elif severity == "Severe":
        h_min, h_max = max(18, hours), max(30, hours + 6)
    else:
        h_min, h_max = max(6, hours - 2), hours + 2

    return {
        "severity": severity,
        "hours_min": h_min,
        "hours_max": h_max
    }

# -------------------------
# AI VISION (stub – auto-on)
# -------------------------
def ai_vision_analysis(files: List[UploadFile]) -> Dict[str, Any]:
    """
    AI AUTO-ON
    Stub now, real OpenAI Vision drops in here.
    """

    return {
        "severity": "Moderate",
        "confidence": "Medium",
        "summary": "Front-left impact visible from driver perspective.",
        "damaged_areas": [
            "Front bumper (driver side)",
            "Left fender",
            "Left headlight"
        ],
        "operations": [
            "Replace bumper",
            "Repair fender",
            "Replace headlight"
        ],
        "risk_note": "Hidden damage is common in front-corner impacts."
    }

# -------------------------
# Pages (HTML inline = zero missing file risk)
# -------------------------
def render_upload(shop_name: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>{shop_name} — AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body class="page">
  <div class="card">
    <div class="title">{shop_name}</div>
    <div class="subtitle">Upload up to 3 photos for an AI repair estimate.</div>

    <form id="form">
      <input type="file" name="photos" accept="image/*" multiple required>
      <button class="cta">Analyze</button>
    </form>

    <script>
      const form = document.getElementById("form");
      form.onsubmit = async (e) => {{
        e.preventDefault();
        const fd = new FormData(form);
        const r = await fetch("/estimate/api", {{ method: "POST", body: fd }});
        const j = await r.json();
        window.location.href = "/estimate/result?id=" + j.estimate_id;
      }};
    </script>
  </div>
</body>
</html>
"""

def render_result(d: Dict[str, Any]) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body class="page">
  <div class="card">
    <div class="title">AI Estimate</div>
    <div class="pill">{d['severity']} • {d['confidence']} confidence</div>

    <p>{d['summary']}</p>

    <ul>{"".join(f"<li>{x}</li>" for x in d['damaged_areas'])}</ul>
    <ul>{"".join(f"<li>{x}</li>" for x in d['operations'])}</ul>

    <div>Labour: {d['hours_min']} – {d['hours_max']} hours</div>

    <div class="big">{d['cost_min']} – {d['cost_max']}</div>

    <div class="warning">
      <strong>Possible final repair cost may be higher</strong><br>
      {d['risk_note']}
    </div>

    <a class="backlink" href="/estimate">← Start over</a>
  </div>
</body>
</html>
"""

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return RedirectResponse("/estimate")

@app.get("/estimate", response_class=HTMLResponse)
def estimate_page():
    shop = SHOP_CONFIGS["miss"]
    return HTMLResponse(render_upload(shop["name"]))

@app.post("/estimate/api")
async def estimate_api(
    photos: List[UploadFile] = File(...)
):
    ai = ai_vision_analysis(photos)
    rules = rule_engine(ai)

    labor_rate = SHOP_CONFIGS["miss"]["labor_rate"]
    cost_min = rules["hours_min"] * labor_rate
    cost_max = rules["hours_max"] * labor_rate

    eid = str(uuid.uuid4())
    ESTIMATES[eid] = {
        **ai,
        **rules,
        "cost_min": money(cost_min),
        "cost_max": money(cost_max)
    }

    return JSONResponse({"estimate_id": eid})

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    d = ESTIMATES.get(id)
    if not d:
        return RedirectResponse("/estimate")
    return HTMLResponse(render_result(d))
