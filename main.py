import uuid
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -------------------------------------------------
# Static files
# -------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Shop configuration
# -------------------------------------------------
SHOP_CONFIGS = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110
    }
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def money(n: int) -> str:
    return f"${n:,}"

# -------------------------------------------------
# RULE ENGINE (OVERRIDES AI OPTIMISM)
# -------------------------------------------------
def rule_engine(ai: Dict[str, Any]) -> Dict[str, int]:
    OP_HOURS = {
        "Replace bumper": 4,
        "Repair bumper": 3,
        "Replace fender": 5,
        "Repair fender": 3,
        "Replace headlight": 2,
        "Repair hood": 4,
        "Inspect suspension": 3,
    }

    hours = sum(OP_HOURS.get(op, 2) for op in ai["operations"])

    severity = ai["severity"]

    if severity == "Severe":
        return {"hours_min": max(20, hours), "hours_max": max(35, hours + 6)}
    if severity == "Minor":
        return {"hours_min": max(2, hours - 1), "hours_max": hours + 1}

    return {"hours_min": max(6, hours - 2), "hours_max": hours + 2}

# -------------------------------------------------
# AI VISION (AUTO-ON STUB)
# -------------------------------------------------
def ai_vision_analysis(files: List[UploadFile]) -> Dict[str, Any]:
    # THIS WILL BE REPLACED WITH REAL GPT-4.1 VISION
    return {
        "severity": "Moderate",
        "confidence": "Medium",
        "summary": "Front-left impact detected from driver perspective.",
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

# -------------------------------------------------
# Pages
# -------------------------------------------------
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
    <div class="subtitle">Upload 1–3 photos for an AI repair estimate.</div>

    <form id="form">
      <input type="file" name="photos" accept="image/*" multiple required>
      <button class="cta">Analyze</button>
    </form>

    <script>
      document.getElementById("form").onsubmit = async (e) => {{
        e.preventDefault();
        const fd = new FormData(e.target);
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

# -------------------------------------------------
# ROUTES (THIS IS WHAT FIXES EVERYTHING)
# -------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/estimate")

@app.get("/quote", include_in_schema=False)
def legacy_quote():
    return RedirectResponse("/estimate")

@app.get("/quote/{shop}", include_in_schema=False)
def legacy_quote_shop(shop: str):
    return RedirectResponse("/estimate")

@app.get("/estimate", response_class=HTMLResponse)
def estimate_page():
    return HTMLResponse(render_upload(SHOP_CONFIGS["miss"]["name"]))

@app.post("/estimate/api")
async def estimate_api(
    photos: List[UploadFile] = File(...)
):
    ai = ai_vision_analysis(photos)
    rules = rule_engine(ai)

    rate = SHOP_CONFIGS["miss"]["labor_rate"]
    cost_min = rules["hours_min"] * rate
    cost_max = rules["hours_max"] * rate

    eid = str(uuid.uuid4())
    ESTIMATES[eid] = {
        **ai,
        **rules,
        "cost_min": money(cost_min),
        "cost_max": money(cost_max),
    }

    return JSONResponse({"estimate_id": eid})

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    d = ESTIMATES.get(id)
    if not d:
        return RedirectResponse("/estimate")
    return HTMLResponse(render_result(d))
