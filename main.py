import os
import json
import uuid
import base64
from typing import Dict, Any, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

app = FastAPI()

# Serve static files (CSS + logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------
# Shop config (add more shops here)
# NOTE: Keep Miss at 110 so your current $ ranges don't change.
# You can later change per-shop rates without touching UI.
# --------------------------------------------------------------------
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rates": {  # CAD/hour
            "body": 110,
            "paint": 110,
            "mechanical": 110,
            "diagnostic": 110,
        },
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
        "labor_rates": SHOP_CONFIGS["miss"]["labor_rates"],
    }
    return canonical, fallback


# --------------------------------------------------------------------
# "AI Auto On" — OpenAI Vision (LIVE) with safe fallback
# --------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")


def _data_url_for_image(content: bytes, content_type: str) -> str:
    b64 = base64.b64encode(content).decode("utf-8")
    # Ensure a valid mime
    mime = content_type if content_type else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def ai_analyze_damage(
    image_bytes: bytes,
    image_mime: str,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      severity: Minor|Moderate|Severe
      confidence: Low|Medium|High
      summary: str
      damaged_areas: [str]  (use driver POV e.g. "Front left bumper")
      operations: [str]     (e.g. "Replace front left bumper cover")
      notes: str (optional)
    """
    if _openai_client is None:
        return None

    img_url = _data_url_for_image(image_bytes, image_mime)

    system = (
        "You are an expert collision estimator. "
        "Use DRIVER'S POV for left/right (driver seated facing forward). "
        "Be concise and practical like a body shop intake estimate."
    )

    # Keep output strictly JSON so parsing is reliable.
    user = (
        "Analyze this vehicle damage photo and return ONLY JSON with keys:\n"
        "{\n"
        '  "severity": "Minor|Moderate|Severe",\n'
        '  "confidence": "Low|Medium|High",\n'
        '  "summary": "1-2 sentences",\n'
        '  "damaged_areas": ["..."],\n'
        '  "operations": ["..."]\n'
        "}\n"
        "Rules:\n"
        "- damaged_areas should include location when possible (front/rear/left/right) using DRIVER POV.\n"
        "- operations should be realistic shop actions (replace/repair/refinish/align/scan/inspect).\n"
        "- If you are unsure about side, do NOT guess; omit left/right.\n"
    )

    try:
        resp = _openai_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()

        # Some models wrap JSON in ```; strip safely.
        if text.startswith("```"):
            text = text.strip("`")
            # If it starts like "json\n{...}"
            if "\n" in text:
                text = text.split("\n", 1)[1].strip()

        data = json.loads(text)

        # Basic validation / normalization
        if not isinstance(data, dict):
            return None

        severity = str(data.get("severity", "")).title()
        confidence = str(data.get("confidence", "")).title()
        summary = str(data.get("summary", "")).strip()

        damaged_areas = data.get("damaged_areas", [])
        operations = data.get("operations", [])

        if not isinstance(damaged_areas, list) or not all(isinstance(x, str) for x in damaged_areas):
            damaged_areas = []
        if not isinstance(operations, list) or not all(isinstance(x, str) for x in operations):
            operations = []

        if severity not in {"Minor", "Moderate", "Severe"}:
            return None
        if confidence not in {"Low", "Medium", "High"}:
            confidence = "Medium"
        if not summary:
            summary = "Visible body damage detected. Further inspection recommended."

        return {
            "severity": severity,
            "confidence": confidence,
            "summary": summary,
            "damaged_areas": damaged_areas[:8],
            "operations": operations[:10],
        }
    except Exception:
        # Never crash the request; just fall back
        return None


# --------------------------------------------------------------------
# Estimation logic: fallback severity -> plan -> operation-hours -> cost
# --------------------------------------------------------------------
def infer_severity(filename: str, size_bytes: int) -> Tuple[str, str]:
    """
    Fallback heuristic (used only if AI fails or key missing).
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


def severity_plan(severity: str) -> Tuple[str, List[str], List[str]]:
    """
    Fallback plan if AI isn't available.
    Returns (summary, damaged_areas, operations)
    """
    if severity == "Minor":
        summary = "Light cosmetic damage likely. Further inspection recommended."
        damaged = ["Bumper", "Fender"]
        ops = ["Repair bumper", "Refinish bumper", "Repair fender"]
        return summary, damaged, ops

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
        return summary, damaged, ops

    # Moderate (default)
    summary = "Visible body damage detected. Further inspection recommended."
    damaged = ["Bumper", "Fender", "Headlight"]
    ops = ["Replace bumper", "Repair fender", "Replace headlight"]
    return summary, damaged, ops


def risk_note_for(severity: str) -> str:
    if severity == "Severe":
        return "Hidden damage is common in front-corner impacts."
    if severity == "Minor":
        return "Final cost may vary after in-person inspection."
    return "Hidden damage is common in moderate impacts."


def money_fmt(n: int) -> str:
    return f"${n:,}"


# --------------------------------------------------------------------
# Operation-based labour hours (real estimator-style heuristic)
# --------------------------------------------------------------------
OP_CATALOG: Dict[str, Dict[str, Any]] = {
    # action keywords -> (min,max) hours + category for rate selection
    "replace bumper": {"hours": (4, 7), "cat": "body"},
    "repair bumper": {"hours": (2, 4), "cat": "body"},
    "refinish bumper": {"hours": (2, 4), "cat": "paint"},

    "replace fender": {"hours": (5, 8), "cat": "body"},
    "repair fender": {"hours": (3, 6), "cat": "body"},
    "refinish fender": {"hours": (2, 4), "cat": "paint"},

    "replace headlight": {"hours": (1, 2), "cat": "body"},
    "replace taillight": {"hours": (1, 2), "cat": "body"},

    "repair hood": {"hours": (3, 6), "cat": "body"},
    "replace hood": {"hours": (5, 8), "cat": "body"},
    "refinish hood": {"hours": (2, 4), "cat": "paint"},

    "inspect suspension": {"hours": (2, 4), "cat": "mechanical"},
    "align": {"hours": (1, 2), "cat": "mechanical"},

    "scan": {"hours": (1, 2), "cat": "diagnostic"},
    "calibrate": {"hours": (2, 4), "cat": "diagnostic"},
}


def _normalize_op(op: str) -> str:
    return " ".join((op or "").lower().replace("-", " ").split())


def estimate_hours_from_operations(operations: List[str], severity: str) -> Tuple[int, int]:
    """
    Sum per-operation hours. Add a small buffer depending on severity.
    """
    base_min = 0.0
    base_max = 0.0

    for op in operations:
        key = _normalize_op(op)

        # Match by contains (flexible)
        matched = None
        for k in OP_CATALOG.keys():
            if k in key:
                matched = OP_CATALOG[k]
                break

        if matched:
            hmin, hmax = matched["hours"]
        else:
            # Unknown op: conservative small allowance
            hmin, hmax = (1, 2)

        base_min += hmin
        base_max += hmax

    # Buffers (teardown / hidden damage / admin overlap)
    if severity == "Minor":
        buffer_min, buffer_max = (0.5, 1.5)
    elif severity == "Severe":
        buffer_min, buffer_max = (2.0, 6.0)
    else:
        buffer_min, buffer_max = (1.0, 3.0)

    total_min = max(1, int(round(base_min + buffer_min)))
    total_max = max(total_min + 1, int(round(base_max + buffer_max)))

    return total_min, total_max


def estimate_cost_from_operations(
    operations: List[str],
    hours_min: int,
    hours_max: int,
    labor_rates: Dict[str, int],
) -> Tuple[int, int]:
    """
    Simple blended-rate cost using operation categories.
    To preserve your current ranges, this still aligns closely to hours * 110.
    """
    # If we can categorize operations, compute weighted average rate.
    cats: List[str] = []
    for op in operations:
        key = _normalize_op(op)
        cat = None
        for k, meta in OP_CATALOG.items():
            if k in key:
                cat = meta["cat"]
                break
        cats.append(cat or "body")

    if not cats:
        avg_rate = int(labor_rates.get("body", 110))
    else:
        total = 0
        for c in cats:
            total += int(labor_rates.get(c, labor_rates.get("body", 110)))
        avg_rate = int(round(total / len(cats)))

    return hours_min * avg_rate, hours_max * avg_rate


# --------------------------------------------------------------------
# Pages (UI unchanged)
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

    # 1) AI Auto On (LIVE)
    ai = ai_analyze_damage(content, photo.content_type or "image/jpeg")

    if ai:
        severity = ai["severity"]
        confidence = ai["confidence"]
        summary = ai["summary"]
        damaged_areas = ai["damaged_areas"] or []
        operations = ai["operations"] or []
    else:
        # 2) Fallback heuristic
        severity, confidence = infer_severity(photo.filename or "", len(content))
        summary, damaged_areas, operations = severity_plan(severity)

    # 3) Operation-based labour hours
    hours_min, hours_max = estimate_hours_from_operations(operations, severity)

    # 4) Per-shop labor rates (kept at 110 for Miss by config)
    labor_rates = cfg.get("labor_rates", SHOP_CONFIGS["miss"]["labor_rates"])
    cost_min_int, cost_max_int = estimate_cost_from_operations(
        operations=operations,
        hours_min=hours_min,
        hours_max=hours_max,
        labor_rates=labor_rates,
    )

    risk_note = risk_note_for(severity)

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
        "cost_min": money_fmt(cost_min_int),
        "cost_max": money_fmt(cost_max_int),
        "risk_note": risk_note,
    }

    return JSONResponse({"estimate_id": estimate_id})


@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return RedirectResponse(url="/quote?shop_id=miss")
    return HTMLResponse(render_result(data))
