import asyncio
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -----------------------------
# Static files
# -----------------------------
# Required:
# /static/style.css
# /static/logo.png
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Shops (demo-safe)
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
# STEP 1 — LANDING / COMMITMENT
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

<div class="shell">
  <div class="card hero">

    <img src="/static/logo.png" class="logo" alt="{shop_name}"
         onerror="this.style.display='none'"/>

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">
      Upload photos to get a fast AI repair estimate.
    </div>

    <a href="/estimate?shop_key={shop_key}" class="cta">
      Start Estimate
    </a>

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
# STEP 2 — UPLOAD + LOADER
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

<div class="shell">
  <div class="card hero">

    <img src="/static/logo.png" class="logo" alt="{shop_name}"
         onerror="this.style.display='none'"/>

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">
      Upload a photo of the damage to receive a quick repair estimate.
    </div>

    <form id="estimateForm" class="upload-form">
      <div class="file-wrap">
        <input id="photoInput" type="file" name="photo" accept="image/*" required/>
        <input type="hidden" name="shop_key" value="{shop_key}"/>
      </div>

      <button id="submitBtn" class="cta" type="submit">
        Get Estimate
      </button>

      <div class="fineprint">Usually takes 5–10 seconds.</div>
    </form>

    <div id="loader" class="analyzer" style="display:none;">
      <div class="steps" id="loaderText">Analyzing vehicle damage…</div>
      <div class="pill2" id="loaderSub">Reviewing uploaded photos</div>

      <div class="progress">
        <div class="bar">
          <div id="progressBar" class="fill"></div>
        </div>
      </div>
    </div>

    <div id="errorBox" class="warning" style="display:none;"></div>

    <a class="backlink" href="/quote?shop_id={shop_key}">← Back</a>

  </div>
</div>

<script>
(function() {{
  const form = document.getElementById("estimateForm");
  const btn = document.getElementById("submitBtn");
  const loader = document.getElementById("loader");
  const loaderText = document.getElementById("loaderText");
  const loaderSub = document.getElementById("loaderSub");
  const bar = document.getElementById("progressBar");
  const errorBox = document.getElementById("errorBox");
  const photoInput = document.getElementById("photoInput");

  let locked = false;

  const steps = [
    {{ text: "Analyzing vehicle damage…", sub: "Reviewing uploaded photos", width: 25, delay: 2000 }},
    {{ text: "Identifying damaged panels", sub: "Assessing severity and repair type", width: 55, delay: 3000 }},
    {{ text: "Calculating repair estimate", sub: "Labor, materials, and parts", width: 85, delay: 3000 }},
    {{ text: "Finalizing estimate", sub: "", width: 100, delay: 1000 }},
  ];

  function runLoader() {{
    let t = 0;
    steps.forEach(s => {{
      setTimeout(() => {{
        loaderText.textContent = s.text;
        loaderSub.textContent = s.sub;
        bar.style.width = s.width + "%";
      }}, t);
      t += s.delay;
    }});
    return t;
  }}

  form.addEventListener("submit", async (e) => {{
    e.preventDefault();
    if (locked || !photoInput.files.length) return;

    locked = true;
    btn.classList.add("disabled");
    btn.textContent = "Analyzing photos…";
    loader.style.display = "block";
    errorBox.style.display = "none";

    const total = runLoader();
    const formData = new FormData(form);

    let id = null;
    try {{
      const res = await fetch("/api/estimate", {{
        method: "POST",
        body: formData
      }});
      if (!res.ok) throw new Error();
      const data = await res.json();
      id = data.estimate_id;
    }} catch {{
      errorBox.style.display = "block";
      errorBox.innerHTML = "<strong>Issue</strong> Unable to generate estimate.";
      btn.textContent = "Get Estimate";
      btn.classList.remove("disabled");
      locked = false;
      return;
    }}

    setTimeout(() => {{
      window.location.href = "/estimate/result?id=" + id;
    }}, total);
  }});
}})();
</script>

</body>
</html>
"""

# ============================================================
# STEP 3 — ESTIMATE RESULT (PREMIUM)
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

    <img src="/static/logo.png" class="logo" alt="SimpleQuotez"/>

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
      <div class="subtle">
        Photo-based preliminary range. Final scope and pricing confirmed after teardown.
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

    <div class="subtle bottom">
      This estimate is early guidance only. Final repairability, ADAS calibration,
      structural, cooling-system, suspension, or electronic damage may be discovered
      during teardown and in-person inspection.
    </div>

    <a class="backlink" href="/quote?shop_id=miss">← Start over</a>

  </div>
</div>

</body>
</html>
"""

# ============================================================
# ESTIMATE INTELLIGENCE (STRUCTURED)
# ============================================================
@app.post("/api/estimate")
async def estimate_api(
    photo: UploadFile = File(...),
    shop_key: Optional[str] = None
):
    await asyncio.sleep(0.8)

    estimate_id = str(uuid.uuid4())

    ESTIMATES[estimate_id] = {
        "severity": "Severe",
        "confidence": "High",
        "summary": (
            "Significant damage to the front-left area indicates potential structural involvement. "
            "ADAS calibration may be required after repairs."
        ),
        "damaged_areas": [
            "Front bumper",
            "Left fender",
            "Left headlight",
            "Hood",
            "Front suspension components"
        ],
        "operations": [
            "Replace front bumper",
            "Replace left fender",
            "Replace left headlight",
            "Repair hood",
            "Inspect suspension and alignment"
        ],
        "cost_min": 5220,
        "cost_max": 7820,
        "body_hours": 23.0,
        "paint_hours": 6.5,
        "materials": 250,
        "parts": 2680,
        "risk_note": (
            "Front-corner impacts often involve hidden damage "
            "(absorbers, brackets, radiator support, sensors, alignment, calibration)."
        )
    }

    return JSONResponse({"estimate_id": estimate_id})
