import asyncio
import uuid
from typing import Optional, List

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
# STEP 1 — LANDING
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
# STEP 2 — UPLOAD
# ============================================================
@app.get("/estimate", response_class=HTMLResponse)
def upload_page(shop_key: Optional[str] = None):
    key = shop_key or "miss"
    shop_name = resolve_shop(key)
    return HTMLResponse(render_upload(shop_name, key))

def render_upload(shop_name: str, shop_key: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{shop_name} – Upload</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>

<div class="page">
  <div class="card hero">

    <img src="/static/logo.png" class="logo" alt="{shop_name}" onerror="this.style.display='none'"/>

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">Upload a photo of the damage to receive a quick repair estimate.</div>

    <form id="estimateForm" class="upload-form">
      <input id="photoInput" type="file" name="photo" accept="image/*" required/>
      <input type="hidden" name="shop_key" value="{shop_key}"/>

      <button id="submitBtn" class="cta" type="submit">Get Estimate</button>
      <div class="fineprint">Usually takes 5–10 seconds.</div>
    </form>

    <div id="loader" class="analyzer" style="display:none;">
      <div id="loaderText">Analyzing vehicle damage…</div>
      <div id="loaderSub">Reviewing uploaded photos</div>
      <div class="progress"><div id="progressBar" class="fill"></div></div>
    </div>

    <a class="backlink" href="/quote?shop_id={shop_key}">← Back</a>

  </div>
</div>

<script>
(function() {{
  const form = document.getElementById("estimateForm");
  const btn = document.getElementById("submitBtn");
  const loader = document.getElementById("loader");
  const bar = document.getElementById("progressBar");

  const steps = [25, 55, 85, 100];
  let locked = false;

  form.addEventListener("submit", async (e) => {{
    e.preventDefault();
    if (locked) return;
    locked = true;

    btn.textContent = "Analyzing photos…";
    loader.style.display = "block";

    let i = 0;
    const interval = setInterval(() => {{
      bar.style.width = steps[i] + "%";
      i++;
      if (i >= steps.length) clearInterval(interval);
    }}, 1200);

    const formData = new FormData(form);
    const res = await fetch("/api/estimate", {{ method: "POST", body: formData }});
    const data = await res.json();

    setTimeout(() => {{
      window.location.href = "/estimate/result?id=" + data.estimate_id;
    }}, 5000);
  }});
}})();
</script>

</body>
</html>
"""

# ============================================================
# STEP 3 — RESULT
# ============================================================
ESTIMATES = {}

@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return HTMLResponse("<h3>Estimate not found</h3>")

    def bullets(items: List[str]) -> str:
        return "".join(f"<li>{i}</li>" for i in items)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Estimate</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>

<div class="page">
  <div class="card hero">

    <img src="/static/logo.png" class="logo"/>

    <h1 class="title">AI Estimate</h1>

    <div class="pill">{data['severity']} • {data['confidence']} confidence</div>

    <div class="summary">{data['summary']}</div>

    <ul>{bullets(data['damaged_areas'])}</ul>
    <ul>{bullets(data['operations'])}</ul>

    <div class="estimate-hours">
    Labour: {{ labour_hours_min }} – {{ labour_hours_max }} hours 
    </div>

   
    <div class="big">${data['cost_min']:,} – ${data['cost_max']:,}</div>

    <div class="warning">
      <strong>Possible final repair cost may be higher</strong><br/>
      {data['risk_note']}
    </div>

    <a class="backlink" href="/quote?shop_id=miss">← Start over</a>

  </div>
</div>

</body>
</html>
"""

# ============================================================
# API
# ============================================================
@app.post("/api/estimate")
async def estimate_api(photo: UploadFile = File(...)):
    # Read uploaded photo (not analyzed yet)
    await photo.read()

    estimate_id = str(uuid.uuid4())

    # -----------------------------
    # Severity → Parts → Hours rules
    # -----------------------------
    severity = "Moderate"

    if severity == "Minor":
        confidence = "High"
        summary = "Minor cosmetic damage detected. Repair is likely straightforward."
        damaged_areas = ["Bumper"]
        operations = ["Repair bumper"]
        hours_min, hours_max = 2, 4
        risk_note = "Minor hidden damage is possible but unlikely."

    elif severity == "Severe":
        confidence = "High"
        summary = "Severe damage detected with possible structural involvement."
        damaged_areas = [
            "Bumper",
            "Fender",
            "Headlight",
            "Suspension",
            "Structural components"
        ]
        operations = [
            "Replace bumper",
            "Replace fender",
            "Replace headlight",
            "Inspect suspension",
            "Structural alignment"
        ]
        hours_min, hours_max = 20, 40
        risk_note = "Final repair cost may increase after teardown."

    else:
        # Moderate (default)
        confidence = "Medium"
        summary = "Visible body damage detected. Further inspection recommended."
        damaged_areas = ["Bumper", "Fender", "Headlight"]
        operations = [
            "Replace bumper",
            "Repair fender",
            "Replace headlight"
        ]
        hours_min, hours_max = 8, 14
        risk_note = "Hidden damage is common in moderate impacts."

    # -----------------------------
    # Cost calculation
    # -----------------------------
    LABOR_RATE = 110  # $/hour
    cost_min = hours_min * LABOR_RATE
    cost_max = hours_max * LABOR_RATE

    # -----------------------------
    # Store estimate
    # -----------------------------
    ESTIMATES[estimate_id] = {
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "damaged_areas": damaged_areas,
        "operations": operations,
        "labour_hours_min": hours_min,
        "labour_hours_max": hours_max,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "risk_note": risk_note
    }

    return JSONResponse({"estimate_id": estimate_id})
