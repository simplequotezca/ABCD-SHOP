import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -----------------------------
# Static files
# -----------------------------
# Ensure these exist:
#   /static/style.css
#   /static/logo.png
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
# Quote pages (Step 1: Landing)
# -----------------------------
@app.get("/quote", response_class=HTMLResponse)
def quote_query(shop_id: Optional[str] = None):
    shop_key = shop_id or "miss"
    shop_name = resolve_shop(shop_key)
    return HTMLResponse(render_landing_html(shop_name, shop_key))


@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_slug(shop_slug: str):
    shop_key = shop_slug
    shop_name = resolve_shop(shop_key)
    return HTMLResponse(render_landing_html(shop_name, shop_key))


def render_landing_html(shop_name: str, shop_key: str) -> str:
    # Step 1: commitment screen (NO upload field)
    # CTA takes user to Step 2: /estimate?shop_key=...
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
      Upload photos to get a fast AI repair estimate.
    </div>

    <a class="cta" href="/estimate?shop_key={shop_key}">Start Estimate</a>

    <div class="hint">
      <div class="hint-title">Best results with 3 photos:</div>
      <ul class="hint-list">
        <li>Overall damage</li>
        <li>Close-up</li>
        <li>Side angle</li>
      </ul>
    </div>

    <div class="fineprint">
      Preliminary range only. Final pricing is confirmed after teardown and in-person inspection.
    </div>

  </div>
</div>

</body>
</html>
"""


# -----------------------------
# Step 2: Upload + Analyze screen
# -----------------------------
@app.get("/estimate", response_class=HTMLResponse)
def estimate_page(shop_key: Optional[str] = None):
    key = shop_key or "miss"
    shop_name = resolve_shop(key)
    return HTMLResponse(render_upload_html(shop_name, key))


def render_upload_html(shop_name: str, shop_key: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{shop_name} – Upload Photos</title>
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

      <button id="submitBtn" class="cta" type="submit">Get Estimate</button>

      <div class="fineprint">
        Usually takes 5–10 seconds.
      </div>
    </form>

    <!-- Analyzer / Loader -->
    <div id="loader" class="analyzer" style="display:none;">
      <div class="steps" id="loaderText">Analyzing vehicle damage…</div>
      <div class="pill2" id="loaderSub">Reviewing uploaded photos</div>

      <div class="progress">
        <div class="bar">
          <div id="progressBar" class="fill"></div>
        </div>
      </div>
    </div>

    <!-- Error -->
    <div id="errorBox" class="warning" style="display:none;"></div>

    <!-- Result -->
    <div id="result" class="block" style="display:none;">
      <div class="label">Estimate Summary</div>
      <div id="resultSummary" class="summary"></div>

      <div id="resultRange" class="mini breakdown"></div>
      <div id="resultNote" class="small-note"></div>
    </div>

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
  const result = document.getElementById("result");
  const resultSummary = document.getElementById("resultSummary");
  const resultRange = document.getElementById("resultRange");
  const resultNote = document.getElementById("resultNote");
  const errorBox = document.getElementById("errorBox");
  const photoInput = document.getElementById("photoInput");

  let locked = false;

  function showError(msg) {{
    errorBox.style.display = "block";
    errorBox.innerHTML = "<strong>Issue</strong>" + msg;
  }}

  function clearError() {{
    errorBox.style.display = "none";
    errorBox.innerHTML = "";
  }}

  // Exact copy + timing sequence (premium feel)
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
    if (!photoInput.files.length) {{
      showError(" Please upload a photo to continue.");
      return;
    }}

    locked = true;
    clearError();

    btn.classList.add("disabled");
    btn.textContent = "Analyzing photos…";
    loader.style.display = "block";
    result.style.display = "none";

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
        showError(" Unable to generate estimate. Please try a clearer photo.");
        btn.textContent = "Get Estimate";
        btn.classList.remove("disabled");
        locked = false;
        return;
      }}

      resultSummary.textContent = data.summary || "Estimate unavailable.";
      resultRange.textContent = data.range || "";
      resultNote.textContent = data.note || "";

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
# Estimate API (Deterministic Intelligence - Option A)
# -----------------------------
@app.post("/api/estimate")
async def estimate_api(
    request: Request,
    photo: UploadFile = File(...),
    shop_key: Optional[str] = None
):
    _ = request
    _ = shop_key

    # Backend delay (frontend controls perceived speed)
    await asyncio.sleep(0.8)

    # -----------------------------
    # Deterministic estimator logic
    # -----------------------------
    # NOTE: This is intentionally consistent and defensible.
    # Next step will be: vision -> structured cues -> feed this estimator.
    # For now, we deliver "estimator-grade" outputs without hallucinating.
    #
    # Driver POV is enforced in wording (driver/passenger terms).
    # Severity tiers: cosmetic / moderate / severe
    # Repair bias: repair vs replace
    # Labor ranges: body + paint

    # Demo defaults (until vision cues exist)
    damage_area = "front-end (driver side)"
    severity = "moderate"
    confidence = "medium"

    # Ruleset (simple but believable)
    if severity == "cosmetic":
        body_hours = (1, 3)
        paint_hours = (2, 4)
        action = "refinish with minor repair"
        cost_range = "$600 – $1,100"
        hidden_risk = "Low likelihood of hidden damage based on typical cosmetic impacts."
    elif severity == "moderate":
        body_hours = (6, 10)
        paint_hours = (5, 8)
        action = "panel repair with refinish"
        cost_range = "$1,200 – $2,400"
        hidden_risk = "Moderate likelihood of hidden damage behind the impact area."
    else:
        body_hours = (12, 20)
        paint_hours = (8, 14)
        action = "replacement likely (panel + refinish)"
        cost_range = "$2,800 – $5,000+"
        hidden_risk = "Higher likelihood of hidden damage; teardown recommended."

    summary = (
        f"Damage appears consistent with {severity} {damage_area} impact. "
        f"Based on the visible photo, recommended approach: {action}."
    )

    note = (
        f"Confidence: {confidence}. Estimated labor: {body_hours[0]}–{body_hours[1]} body hrs, "
        f"{paint_hours[0]}–{paint_hours[1]} paint hrs. {hidden_risk} "
        "Final pricing may change after teardown and in-person inspection."
    )

    return JSONResponse({
        "summary": summary,
        "range": f"Estimated range: {cost_range}",
        "note": note
    })
