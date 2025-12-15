import asyncio
import uuid
import hashlib
from typing import Optional, List, Dict, Any, Tuple

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
# SEVERITY → PARTS → HOURS LOGIC (DROP-IN, DETERMINISTIC)
# ============================================================

def _seed_from_bytes(photo_bytes: bytes) -> int:
    # Deterministic seed from uploaded photo content
    h = hashlib.sha256(photo_bytes).hexdigest()
    return int(h[:8], 16)

def _pick(seed: int, options: List[str], idx: int = 0) -> str:
    if not options:
        return ""
    return options[(seed + idx) % len(options)]

def infer_signals(photo_bytes: bytes) -> Dict[str, bool]:
    """
    V1 (no AI): derive stable "signals" from photo bytes to create believable variability.
    Later: replace this with actual vision inference.
    """
    seed = _seed_from_bytes(photo_bytes)

    # Deterministic pseudo-signals (stable per image)
    # These are NOT claims about the image — they are placeholders to drive consistent variation.
    corner_damage = (seed % 100) < 70
    bumper_damage = (seed // 3 % 100) < 75
    hood_damage = (seed // 7 % 100) < 55
    headlight_damage = (seed // 11 % 100) < 60

    # Higher-risk signals appear less frequently
    wheel_visible = (seed // 13 % 100) < 28
    suspension_visible = (seed // 17 % 100) < 18
    airbag_possible = (seed // 19 % 100) < 12

    return {
        "corner_damage": corner_damage,
        "bumper_damage": bumper_damage,
        "hood_damage": hood_damage,
        "headlight_damage": headlight_damage,
        "wheel_visible": wheel_visible,
        "suspension_visible": suspension_visible,
        "airbag_possible": airbag_possible,
    }

def decide_severity(signals: Dict[str, bool]) -> str:
    # Rule table (monotonic: only upgrades)
    if signals["wheel_visible"] or signals["suspension_visible"] or signals["airbag_possible"]:
        return "Severe"

    if signals["bumper_damage"] and signals["hood_damage"]:
        # Moderate–Severe in your earlier logic; keep severity bucket simple for UI
        return "Severe"

    if signals["corner_damage"] and signals["headlight_damage"]:
        return "Moderate"

    return "Minor"

def confidence_for(severity: str, signals: Dict[str, bool]) -> str:
    # Confidence drops with higher severity (hidden damage risk)
    if severity == "Minor":
        return "High"
    if severity == "Moderate":
        return "Medium–High"
    # Severe
    return "Medium"

def hours_band(severity: str, seed: int) -> Tuple[int, int]:
    # Conservative, realistic ranges
    if severity == "Minor":
        base = 6 + (seed % 3)  # 6–8
        return base, base + 6  # up to ~14
    if severity == "Moderate":
        base = 14 + (seed % 5)  # 14–18
        return base, base + 10  # up to ~28
    # Severe
    base = 26 + (seed % 7)  # 26–32
    return base, base + 12   # up to ~44

def parts_and_ops(severity: str, signals: Dict[str, bool], seed: int) -> Tuple[List[str], List[str]]:
    # Build parts list based on severity + signals
    parts: List[str] = []
    ops: List[str] = []

    def add(part: str, op: str):
        if part not in parts:
            parts.append(part)
        if op not in ops:
            ops.append(op)

    if severity == "Minor":
        add("Bumper", "Repair bumper")
        add("Fender liner / clips", "Replace fasteners/clips")
        if signals["headlight_damage"] and (seed % 100) < 30:
            add("Headlight bracket", "Repair mounting/bracket")

    elif severity == "Moderate":
        add("Bumper", "Replace bumper")
        add("Fender", "Replace fender")
        add("Headlight", "Replace headlight")
        if signals["hood_damage"]:
            add("Hood", "Repair hood")
        else:
            # occasional hood edge blend note without claiming replacement
            if (seed % 100) < 35:
                add("Hood edge", "Blend/align hood edge")

    else:  # Severe
        add("Bumper", "Replace bumper")
        add("Fender", "Replace fender")
        add("Headlight", "Replace headlight")
        add("Hood", "Repair hood")

        # Safety/mechanical layer
        add("Suspension", "Inspect suspension")
        if signals["suspension_visible"] and (seed % 100) < 55:
            add("Steering/suspension component", "Measure/diagnose steering & suspension")

        if signals["airbag_possible"] and (seed % 100) < 60:
            add("SRS system", "Scan SRS / safety systems")

    # Keep order stable (nice UX)
    return parts, ops

def summary_text(severity: str, signals: Dict[str, bool], seed: int) -> str:
    minor_templates = [
        "Light cosmetic damage likely limited to bolt-on panels and fasteners.",
        "Minor visible damage with low likelihood of mechanical involvement.",
        "Surface-level impact detected; repair appears primarily cosmetic."
    ]
    moderate_templates = [
        "Moderate front-corner damage affecting exterior panels and lighting components.",
        "Visible panel and lighting damage with elevated risk of hidden alignment issues.",
        "Moderate impact detected; parts replacement and refinishing likely required."
    ]
    severe_templates = [
        "Significant front-corner damage with increased risk of hidden mechanical or structural involvement.",
        "High-impact damage detected; inspection of safety and mechanical components is recommended.",
        "Extensive visible damage with elevated teardown/supplement risk."
    ]

    if severity == "Minor":
        return _pick(seed, minor_templates, 1)
    if severity == "Moderate":
        return _pick(seed, moderate_templates, 2)
    return _pick(seed, severe_templates, 3)

def risk_note(severity: str, signals: Dict[str, bool]) -> str:
    if severity == "Minor":
        return "Final pricing may change if hidden clips/brackets or sensor mounts are affected."
    if severity == "Moderate":
        return "Hidden damage is common behind bumpers and headlights; alignment and sensor checks may add cost."
    return "Hidden damage is common in higher-energy corner impacts; teardown may reveal additional parts and labour."

def cost_from_hours(severity: str, hours_min: int, hours_max: int, seed: int) -> Dict[str, int]:
    """
    V1 cost model (deterministic, conservative).
    - labour = hours * blended_rate
    - parts = severity band
    """
    # Blended labour rate (CAD), stable but slightly varied per image
    blended_rate = 105 + (seed % 11)  # 105–115

    labour_min = hours_min * blended_rate
    labour_max = hours_max * blended_rate

    if severity == "Minor":
        parts_min, parts_max = 450, 1100
    elif severity == "Moderate":
        parts_min, parts_max = 1400, 3200
    else:
        parts_min, parts_max = 2400, 5200

    # Small deterministic variation so totals don't feel copy/paste
    wiggle = (seed % 9) * 35  # 0–280
    parts_min += wiggle
    parts_max += wiggle

    cost_min = labour_min + parts_min
    cost_max = labour_max + parts_max

    return {
        "labour_min": int(labour_min),
        "labour_max": int(labour_max),
        "parts_min": int(parts_min),
        "parts_max": int(parts_max),
        "cost_min": int(cost_min),
        "cost_max": int(cost_max),
    }

def build_estimate(photo_bytes: bytes) -> Dict[str, Any]:
    seed = _seed_from_bytes(photo_bytes)
    signals = infer_signals(photo_bytes)
    severity = decide_severity(signals)
    confidence = confidence_for(severity, signals)
    hmin, hmax = hours_band(severity, seed)
    areas, ops = parts_and_ops(severity, signals, seed)
    summary = summary_text(severity, signals, seed)
    costs = cost_from_hours(severity, hmin, hmax, seed)

    return {
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "damaged_areas": areas,
        "operations": ops,
        "hours_min": hmin,
        "hours_max": hmax,
        "labour_min": costs["labour_min"],
        "labour_max": costs["labour_max"],
        "parts_min": costs["parts_min"],
        "parts_max": costs["parts_max"],
        "cost_min": costs["cost_min"],
        "cost_max": costs["cost_max"],
        "risk_note": risk_note(severity, signals),
    }

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
ESTIMATES: Dict[str, Dict[str, Any]] = {}

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

    <div class="breakdown">
      <div><strong>Estimated labour:</strong> {data['hours_min']}–{data['hours_max']} hrs</div>
      <div><strong>Labour:</strong> ${data['labour_min']:,} – ${data['labour_max']:,}</div>
      <div><strong>Parts:</strong> ${data['parts_min']:,} – ${data['parts_max']:,}</div>
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
    # Simulate "analysis time"
    await asyncio.sleep(1)

    photo_bytes = await photo.read()
    estimate = build_estimate(photo_bytes)

    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = estimate

    return JSONResponse({"estimate_id": estimate_id})
