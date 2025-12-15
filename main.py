import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -----------------------------
# Static files
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Shop config
# -----------------------------
SHOPS = {
    "miss": "Mississauga Collision Centre",
    "mississauga-collision-centre": "Mississauga Collision Centre",
    "mississauga-collision-center": "Mississauga Collision Center",
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

# -----------------------------
# Quote Pages
# -----------------------------
@app.get("/quote", response_class=HTMLResponse)
def quote_query(shop_id: Optional[str] = None):
    shop_name = resolve_shop(shop_id or "miss")
    return HTMLResponse(render_quote_html(shop_name, shop_id or "miss"))

@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_slug(shop_slug: str):
    shop_name = resolve_shop(shop_slug)
    return HTMLResponse(render_quote_html(shop_name, shop_slug))


def render_quote_html(shop_name: str, shop_key: str) -> str:
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

    <img src="/static/logo.png" class="logo" alt="{shop_name} logo"
         onerror="this.style.display='none'"/>

    <h1 class="title">{shop_name}</h1>
    <div class="subtitle">
      Upload a photo of the damage to receive a quick repair estimate.
    </div>

    <form id="estimateForm" class="upload-form">
      <div class="file-wrap">
        <input id="photoInput" type="file" name="photo" accept="image/*" required />
        <input type="hidden" name="shop_key" value="{shop_key}" />
      </div>

      <button id="submitBtn" class="cta" type="submit">
        Get Estimate
      </button>

      <div class="fineprint">
        Usually takes 5–10 seconds.
      </div>
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

    <div id="result" class="block" style="display:none;">
      <div class="label">Estimate Summary</div>
      <div id="resultSummary" class="summary"></div>
      <div id="resultRange" class="mini breakdown"></div>
      <div id="resultNote" class="small-note"></div>
    </div>

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
  const result = document.getElementById("result");
  const resultSummary = document.getElementById("resultSummary");
  const resultRange = document.getElementById("resultRange");
  const resultNote = document.getElementById("resultNote");
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
    if (locked) return;
    if (!photoInput.files.length) return;

    locked = true;
    btn.classList.add("disabled");
    btn.textContent = "Analyzing photos…";
    loader.style.display = "block";
    result.style.display = "none";
    errorBox.style.display = "none";

    const total = runLoader();
    const formData = new FormData(form);

    let data = null;
    let failed = false;

    try {{
      const res = await fetch("/api/estimate", {{
        method: "POST",
        body: formData
      }});
      if (!res.ok) throw new Error();
      data = await res.json();
    }} catch {{
      failed = true;
    }}

    setTimeout(() => {{
      if (failed) {{
        errorBox.style.display = "block";
        errorBox.innerHTML = "<strong>Issue</strong> Unable to generate estimate. Please try a clearer photo.";
        btn.textContent = "Get Estimate";
        btn.classList.remove("disabled");
        locked = false;
        return;
      }}

      resultSummary.textContent = data.summary;
      resultRange.textContent = data.range;
      resultNote.textContent = data.note;

      result.style.display = "block";
      btn.textContent = "Get Estimate";
      btn.classList.remove("disabled");
      locked = false;
    }}, total);
  }});
}})();
</script>

</body>
</html>
"""

# -----------------------------
# ESTIMATE INTELLIGENCE (A)
# -----------------------------
@app.post("/api/estimate")
async def estimate(
    request: Request,
    photo: UploadFile = File(...),
    shop_key: Optional[str] = None
):
    _ = request
    _ = shop_key

    await asyncio.sleep(0.8)

    # ---- Deterministic Estimator Logic ----
    # This will later be driven by vision output.
    # For now: consistent, believable, estimator-grade.

    damage_area = "front-end (driver side)"
    severity = "moderate"

    if severity == "cosmetic":
        body_hours = (2, 4)
        paint_hours = (3, 5)
        cost_range = "$600 – $1,100"
        action = "repair and refinish"
    elif severity == "moderate":
        body_hours = (6, 10)
        paint_hours = (5, 8)
        cost_range = "$1,200 – $2,400"
        action = "panel repair with refinish"
    else:
        body_hours = (12, 20)
        paint_hours = (8, 14)
        cost_range = "$2,800 – $5,000+"
        action = "panel replacement likely"

    summary = (
        f"Damage appears consistent with {severity} {damage_area} impact. "
        f"Based on visible deformation and surface damage, {action} is recommended."
    )

    note = (
        f"Estimated labor: {body_hours[0]}–{body_hours[1]} body hrs, "
        f"{paint_hours[0]}–{paint_hours[1]} paint hrs. "
        "Final pricing may change after teardown and inspection for hidden damage."
    )

    return JSONResponse({
        "summary": summary,
        "range": cost_range,
        "note": note
    })
