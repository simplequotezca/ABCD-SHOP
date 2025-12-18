import uuid
from typing import Dict, Any, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------
# Shop configuration
# --------------------------------------------------------------------
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110,
    }
}

SHOP_ALIASES = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
}

ESTIMATES: Dict[str, Dict[str, Any]] = {}


def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    key = (shop_key or "miss").lower()
    canonical = SHOP_ALIASES.get(key, key)
    cfg = SHOP_CONFIGS.get(canonical)

    if not cfg:
        cfg = SHOP_CONFIGS["miss"]

    return canonical, cfg


# --------------------------------------------------------------------
# Simple estimation logic (AI placeholder)
# --------------------------------------------------------------------
def infer_severity(filename: str, size_bytes: int) -> Tuple[str, str]:
    if size_bytes > 2_500_000:
        return "Severe", "High"
    if size_bytes < 400_000:
        return "Minor", "High"
    return "Moderate", "Medium"


def severity_plan(severity: str):
    if severity == "Severe":
        return (
            "Significant damage with possible structural involvement.",
            ["Front bumper", "Left fender", "Headlight"],
            ["Replace bumper", "Replace fender", "Replace headlight"],
            22,
            38,
            "Hidden damage is common in front-corner impacts.",
        )

    if severity == "Minor":
        return (
            "Light cosmetic damage detected.",
            ["Bumper"],
            ["Repair bumper"],
            3,
            6,
            "Final cost may vary after inspection.",
        )

    return (
        "Visible body damage detected. Further inspection recommended.",
        ["Bumper", "Fender"],
        ["Repair bumper", "Repair fender"],
        8,
        14,
        "Hidden damage is common in moderate impacts.",
    )


def money(n: int) -> str:
    return f"${n:,}"


# --------------------------------------------------------------------
# HTML Renderers
# --------------------------------------------------------------------
def render_landing(shop_key: str, shop_name: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{shop_name}</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <img src="/static/logo.png" class="logo" />
    <h1>{shop_name}</h1>
    <p>Upload photos to get a fast AI repair estimate.</p>
    <a class="cta" href="/estimate?shop_key={shop_key}">Start Estimate</a>
  </div>
</body>
</html>
"""


def render_upload(shop_key: str, shop_name: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Upload</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <h2>{shop_name}</h2>
    <p>Upload 1–3 photos for an AI repair estimate.</p>

    <form method="post" action="/estimate/api" enctype="multipart/form-data">
      <input type="hidden" name="shop_key" value="{shop_key}" />
      <input type="file" name="photo" accept="image/*" required />
      <button class="cta" type="submit">Analyze</button>
    </form>

    <a class="backlink" href="/quote/{shop_key}">← Back</a>
  </div>
</body>
</html>
"""


def render_result(data: Dict[str, Any]) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Estimate</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <h1>AI Estimate</h1>
    <div class="pill">{data["severity"]} • {data["confidence"]} confidence</div>

    <p>{data["summary"]}</p>

    <ul>{"".join(f"<li>{d}</li>" for d in data["damaged_areas"])}</ul>
    <ul>{"".join(f"<li>{o}</li>" for o in data["operations"])}</ul>

    <p>Labour: {data["hours_min"]} – {data["hours_max"]} hours</p>
    <div class="big">{data["cost_min"]} – {data["cost_max"]}</div>

    <div class="warning">{data["risk_note"]}</div>

    <a class="backlink" href="/quote/{data["shop_key"]}">← Start over</a>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------
# Routes
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
async def estimate_api(
    photo: UploadFile = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)

    content = await photo.read()
    severity, confidence = infer_severity(photo.filename, len(content))

    summary, damaged, ops, h_min, h_max, risk = severity_plan(severity)

    rate = cfg["labor_rate"]
    estimate_id = str(uuid.uuid4())

    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "damaged_areas": damaged,
        "operations": ops,
        "hours_min": h_min,
        "hours_max": h_max,
        "cost_min": money(h_min * rate),
        "cost_max": money(h_max * rate),
        "risk_note": risk,
    }

    return RedirectResponse(
        f"/estimate/result?id={estimate_id}",
        status_code=303,
    )


@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return RedirectResponse("/")
    return render_result(data)
