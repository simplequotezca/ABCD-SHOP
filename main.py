from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import uuid, asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# SHOP RESOLUTION
# ============================================================
def resolve_shop(key: str) -> str:
    return "Mississauga Collision Center"

# ============================================================
# STEP 1 — LANDING (LOGO ONLY HERE)
# ============================================================
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

<div class="shell">
  <div class="card hero">

    <img src="/static/logo.png" class="logo" alt="SimpleQuotez"/>

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">
      Upload photos to get a fast AI repair estimate.
    </div>

    <a href="/estimate?shop_key={shop_key}" class="cta">
      Start Estimate
    </a>

    <div class="hint">
      <div class="hint-title">BEST RESULTS WITH 3 PHOTOS:</div>
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

@app.get("/quote", response_class=HTMLResponse)
def landing(shop_id: Optional[str] = None):
    key = shop_id or "miss"
    return HTMLResponse(render_landing(resolve_shop(key), key))

# ============================================================
# STEP 2 — UPLOAD (NO LOGO)
# ============================================================
@app.get("/estimate", response_class=HTMLResponse)
def upload_page(shop_key: Optional[str] = None):
    shop_name = resolve_shop(shop_key or "miss")
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

<div class="shell">
  <div class="card hero">

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">
      Upload a photo of the damage to receive a quick repair estimate.
    </div>

    <form id="estimateForm" class="upload-form">
      <input type="file" name="photo" accept="image/*" required/>
      <button id="submitBtn" class="cta" type="submit">Get Estimate</button>
      <div class="fineprint">Usually takes 5–10 seconds.</div>
    </form>

    <div id="loader" class="analyzer" style="display:none;">
      <div class="steps">Analyzing vehicle damage…</div>
      <div class="pill2">Reviewing uploaded photos</div>
      <div class="progress">
        <div class="bar"><div id="progressBar" class="fill"></div></div>
      </div>
    </div>

  </div>
</div>

<script>
const form = document.getElementById("estimateForm");
const loader = document.getElementById("loader");
form.addEventListener("submit", async e => {{
  e.preventDefault();
  loader.style.display = "block";
  const res = await fetch("/api/estimate", {{
    method: "POST",
    body: new FormData(form)
  }});
  const data = await res.json();
  setTimeout(() => {{
    window.location.href = "/estimate/result?id=" + data.estimate_id;
  }}, 5000);
}});
</script>

</body>
</html>
"""

# ============================================================
# STEP 3 — RESULT (NO LOGO, WITH SUMMARY + LABOR)
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

<div class="shell">
  <div class="card hero">

    <h1 class="title">AI Estimate</h1>

    <div class="pill">
      {data['severity']} • {data['confidence']} confidence
    </div>

    <div class="block">
      <div class="label">Summary</div>
      <div class="summary">{data['summary']}</div>
    </div>

    <div class="block">
      <div class="label">Likely damaged areas</div>
      <ul class="list">{bullets(data['damaged_areas'])}</ul>
    </div>

    <div class="block">
      <div class="label">Recommended operations</div>
      <ul class="list">{bullets(data['operations'])}</ul>
    </div>

    <div class="block">
      <div class="label">Estimate (CAD)</div>
      <div class="text big">
        ${data['cost_min']:,} – ${data['cost_max']:,}
      </div>

      <div class="mini breakdown">
        Body labor: {data['body_hours']}h<br/>
        Paint labor: {data['paint_hours']}h<br/>
        Materials: ${data['materials']}<br/>
        Parts (rough): ${data['parts']}
      </div>
    </div>

    <div class="warning">
      <strong>Possible final repair cost may be higher</strong>
      {data['risk_note']}
    </div>

    <a class="backlink" href="/quote?shop_id=miss">← Start over</a>

  </div>
</div>

</body>
</html>
"""

# ============================================================
# API — ESTIMATE
# ============================================================
@app.post("/api/estimate")
async def estimate_api(photo: UploadFile = File(...)):
    await asyncio.sleep(1)
    eid = str(uuid.uuid4())
    ESTIMATES[eid] = {
        "severity": "Severe",
        "confidence": "High",
        "summary": "Significant front-left damage with possible structural involvement. ADAS calibration may be required.",
        "damaged_areas": ["Front bumper", "Left fender", "Headlight", "Hood", "Suspension"],
        "operations": ["Replace bumper", "Replace fender", "Replace headlight", "Repair hood", "Inspect suspension"],
        "cost_min": 5220,
        "cost_max": 7820,
        "body_hours": 23.0,
        "paint_hours": 6.5,
        "materials": 250,
        "parts": 2680,
        "risk_note": "Front-corner impacts often involve hidden damage."
    }
    return JSONResponse({"estimate_id": eid})
