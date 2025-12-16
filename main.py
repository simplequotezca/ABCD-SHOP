import uuid
from typing import Dict, Any, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files (CSS + logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------
# Shop config (add more shops here)
# NOTE: Keep Miss at 110 so your current $ ranges don't change.
# --------------------------------------------------------------------
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110,  # CAD/hour
    },
}

# Aliases / slugs that should resolve to the canonical shop key above
SHOP_ALIASES: Dict[str, str] = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
    "mississauga_collision_center": "miss",
}

# In-memory store for demo (replace with DB later)
ESTIMATES: Dict[str, Dict[str, Any]] = {}


def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (canonical_shop_key, shop_config)
    """
    key_raw = (shop_key or "miss").strip()
    canonical = SHOP_ALIASES.get(key_raw, key_raw)

    cfg = SHOP_CONFIGS.get(canonical)
    if cfg:
        return canonical, cfg

    # Fallback: accept unknown shops without breaking the UI
    fallback = {
        "name": key_raw.replace("-", " ").title(),
        "labor_rate": SHOP_CONFIGS["miss"]["labor_rate"],
    }
    return canonical, fallback


# --------------------------------------------------------------------
# Estimation logic: severity -> parts -> hours -> cost
# (Heuristic placeholder until you wire real AI vision)
# --------------------------------------------------------------------
def infer_severity(filename: str, size_bytes: int) -> Tuple[str, str]:
    """
    Returns (severity, confidence)
    """
    name = (filename or "").lower()

    severe_kw = ["total", "airbag", "frame", "crush", "severe", "structural", "tow"]
    minor_kw = ["scratch", "scuff", "paint", "chip", "minor", "light"]

    if any(k in name for k in severe_kw) or size_bytes > 2_500_000:
        return "Severe", "High"
    if any(k in name for k in minor_kw) or size_bytes < 350_000:
        return "Minor", "High"
    return "Moderate", "Medium"


def severity_plan(severity: str) -> Tuple[str, List[str], List[str], int, int]:
    """
    Returns (summary, damaged_areas, operations, hours_min, hours_max)
    """
    if severity == "Minor":
        summary = "Light cosmetic damage likely. Further inspection recommended."
        damaged = ["Bumper", "Fender"]
        ops = ["Repair bumper", "Refinish bumper", "Repair fender"]
        return summary, damaged, ops, 3, 6

    if severity == "Severe":
        summary = "Significant damage with possible structural involvement."
        damaged = ["Bumper", "Fender", "Headlight", "Hood", "Suspension"]
        ops = [
            "Replace bumper",
            "Replace fender",
            "Replace headlight",
            "Repair hood",
            "Inspect suspension",
        ]
        return summary, damaged, ops, 22, 38

    # Moderate (default)
    summary = "Visible body damage detected. Further inspection recommended."
    damaged = ["Bumper", "Fender", "Headlight"]
    ops = ["Replace bumper", "Repair fender", "Replace headlight"]
    return summary, damaged, ops, 8, 14


def risk_note_for(severity: str) -> str:
    if severity == "Severe":
        return "Hidden damage is common in front-corner impacts."
    if severity == "Minor":
        return "Final cost may vary after in-person inspection."
    return "Hidden damage is common in moderate impacts."


def money_fmt(n: int) -> str:
    return f"${n:,}"


# --------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------
def render_landing(shop_key: str, shop_name: str) -> str:
    # Logo appears ONLY on the first page.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{shop_name} — AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <img class="logo" src="/static/logo.png" alt="SimpleQuotez" />
    <div class="title">{shop_name}</div>
    <div class="subtitle">Upload photos to get a fast AI repair estimate.</div>

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
      Photo-based preliminary range only. Final pricing is confirmed after teardown and in-person inspection.
    </div>
  </div>
</body>
</html>
"""


def render_upload(shop_key: str, shop_name: str) -> str:
    # No logo on this page.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{shop_name} — Upload</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <div class="title">{shop_name}</div>
    <div class="subtitle">Upload a photo of the damage to receive a quick repair estimate.</div>

    <form id="estimateForm">
      <input type="hidden" name="shop_key" value="{shop_key}" />
      <input type="file" name="photo" accept="image/*" required />
      <button class="cta" id="submitBtn" type="submit">Analyze photo</button>
    </form>

    <div class="analyzer" id="analyzer" style="display:none;">
      <div class="subtitle" style="margin-top:12px;">Usually takes 5–10 seconds.</div>
      <div style="margin-top:10px;">
        <div>Analyzing vehicle damage...</div>
        <div>Reviewing uploaded photos</div>
      </div>
      <div class="progress" style="margin-top:12px;">
        <div class="fill" id="fill"></div>
      </div>
    </div>

    <a class="backlink" href="/quote?shop_id={shop_key}">← Back</a>
  </div>

<script>
(function() {{
  const form = document.getElementById('estimateForm');
  const btn = document.getElementById('submitBtn');
  const analyzer = document.getElementById('analyzer');
  const fill = document.getElementById('fill');

  function startProgress() {{
    let p = 8;
    fill.style.width = p + '%';
    const t = setInterval(() => {{
      p = Math.min(96, p + Math.random() * 9);
      fill.style.width = p + '%';
    }}, 500);
    return () => {{
      clearInterval(t);
      fill.style.width = '100%';
    }};
  }}

  form.addEventListener('submit', async (e) => {{
    e.preventDefault();

    btn.disabled = true;
    btn.textContent = 'Analyzing photos...';
    analyzer.style.display = 'block';

    const stop = startProgress();

    try {{
      const fd = new FormData(form);
      const r = await fetch('/estimate/api', {{ method: 'POST', body: fd }});
      if (!r.ok) throw new Error('Estimate failed');
      const data = await r.json();
      stop();
      window.location.href = '/estimate/result?id=' + encodeURIComponent(data.estimate_id);
    }} catch (err) {{
      btn.disabled = false;
      btn.textContent = 'Analyze photo';
      analyzer.style.display = 'none';
      fill.style.width = '0%';
      alert('Something went wrong. Please try again.');
    }}
  }});
}})();
</script>
</body>
</html>
"""


def bullets(items: List[str]) -> str:
    return "\n".join(f"<li>{i}</li>" for i in items)


def render_result(data: Dict[str, Any]) -> str:
    # No logo on this page.
    pill = f"{data['severity']} • {data['confidence']} confidence"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <div class="title">AI Estimate</div>

    <div class="pill">{pill}</div>

    <div style="margin-top:14px; line-height:1.45;">
      {data["summary"]}
    </div>

    <ul style="margin-top:12px;">{bullets(data["damaged_areas"])}</ul>
    <ul style="margin-top:10px;">{bullets(data["operations"])}</ul>

    <div style="margin-top:10px;">
      Labour: {data["labour_hours_min"]} – {data["labour_hours_max"]} hours
    </div>

    <div class="big">{data["cost_min"]} – {data["cost_max"]}</div>

    <div class="warning">
      <strong>Possible final repair cost may be higher</strong><br/>
      {data["risk_note"]}
    </div>

    <a class="backlink" href="/quote?shop_id={data["shop_key"]}">← Start over</a>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/quote?shop_id=miss")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/quote", response_class=HTMLResponse)
def quote(shop_id: str = "miss"):
    shop_key, cfg = resolve_shop(shop_id)
    return HTMLResponse(render_landing(shop_key=shop_key, shop_name=cfg["name"]))


@app.get("/quote/{shop_key}", response_class=HTMLResponse)
def quote_slug(shop_key: str):
    k, cfg = resolve_shop(shop_key)
    return HTMLResponse(render_landing(shop_key=k, shop_name=cfg["name"]))


@app.get("/estimate", response_class=HTMLResponse)
def upload_page(shop_key: str = "miss"):
    k, cfg = resolve_shop(shop_key)
    return HTMLResponse(render_upload(shop_key=k, shop_name=cfg["name"]))


@app.post("/estimate/api")
async def estimate_api(
    photo: UploadFile = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)

    content = await photo.read()
    severity, confidence = infer_severity(photo.filename or "", len(content))

    summary, damaged_areas, operations, hours_min, hours_max = severity_plan(severity)
    risk_note = risk_note_for(severity)

    # ✅ Per-shop labor rate applied here
    labor_rate = int(cfg.get("labor_rate", SHOP_CONFIGS["miss"]["labor_rate"]))

    cost_min = hours_min * labor_rate
    cost_max = hours_max * labor_rate

    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "damaged_areas": damaged_areas,
        "operations": operations,
        "labour_hours_min": hours_min,
        "labour_hours_max": hours_max,
        "cost_min": money_fmt(cost_min),
        "cost_max": money_fmt(cost_max),
        "risk_note": risk_note,
    }

    return JSONResponse({"estimate_id": estimate_id})


@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return RedirectResponse(url="/quote?shop_id=miss")
    return HTMLResponse(render_result(data))
