import os
import uuid
import json
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

# --------------------------------------------------------------------
# AI Auto On
# - If OPENAI_API_KEY exists AND AI_AUTO_ON is not "0", we attempt AI vision.
# - If AI fails (missing lib, bad key, API error), we silently fall back to heuristics.
# --------------------------------------------------------------------
AI_AUTO_ON_DEFAULT = os.getenv("AI_AUTO_ON", "1").strip() not in ("0", "false", "False", "")


def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Returns (canonical_shop_key, shop_config)"""
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
# Side / location helpers (drivers POV labeling)
# --------------------------------------------------------------------
def normalize_location(loc: str) -> str:
    loc = (loc or "").strip().lower()
    if not loc:
        return ""
    # allow common variants
    loc = loc.replace("front ", "front-").replace("rear ", "rear-")
    loc = loc.replace("back", "rear")
    return loc


def infer_location_and_side(filename: str) -> Tuple[str, str]:
    """Best-effort fallback from filename only."""
    n = (filename or "").lower()
    side = ""
    if "left" in n or "driver" in n:
        side = "left"
    elif "right" in n or "passenger" in n:
        side = "right"

    loc = ""
    if "front" in n:
        loc = "front"
    elif "rear" in n or "back" in n:
        loc = "rear"

    # If we have both, combine like front-left
    if loc and side:
        return f"{loc}-{side}", side
    return loc, side


def format_location(location: str, side: str) -> str:
    """Turns 'front-left' + 'left' into 'front-left (driver side)' for Canada/US."""
    location = normalize_location(location)
    side = (side or "").strip().lower()

    if not location and not side:
        return ""

    # If location includes side, extract it
    if location in ("front-left", "rear-left", "front-right", "rear-right"):
        if location.endswith("-left"):
            side = "left"
        elif location.endswith("-right"):
            side = "right"

    label = ""
    if side == "left":
        label = " (driver side)"
    elif side == "right":
        label = " (passenger side)"

    return f"{location}{label}".strip()


# --------------------------------------------------------------------
# Estimation logic: severity -> parts -> operations -> hours -> cost
# --------------------------------------------------------------------
def infer_severity(filename: str, size_bytes: int) -> Tuple[str, str]:
    """Returns (severity, confidence)"""
    name = (filename or "").lower()

    severe_kw = ["total", "airbag", "frame", "crush", "severe", "structural", "tow"]
    minor_kw = ["scratch", "scuff", "paint", "chip", "minor", "light"]

    if any(k in name for k in severe_kw) or size_bytes > 2_500_000:
        return "Severe", "High"
    if any(k in name for k in minor_kw) or size_bytes < 350_000:
        return "Minor", "High"
    return "Moderate", "Medium"


def severity_plan(severity: str) -> Tuple[str, List[str], List[str]]:
    """Returns (summary, damaged_areas, operations)"""
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


# Operation-based labour hour ranges (min,max)
# Keep these roughly aligned with your current UI demo outputs.
OPS_HOURS: Dict[str, Tuple[float, float]] = {
    # Replace
    "Replace bumper": (3.0, 5.0),
    "Replace fender": (4.0, 6.5),
    "Replace headlight": (0.8, 1.8),
    # Repair
    "Repair bumper": (1.5, 3.0),
    "Repair fender": (2.5, 5.0),
    "Repair hood": (3.0, 6.0),
    # Refinish / paint
    "Refinish bumper": (2.0, 3.5),
    # Mechanical / inspection
    "Inspect suspension": (2.0, 4.0),
}

# Severity overhead hours (teardown/measurements/scan/fitment variance, etc.)
SEVERITY_OVERHEAD: Dict[str, Tuple[float, float]] = {
    "Minor": (0.5, 1.5),
    "Moderate": (1.0, 3.0),
    "Severe": (4.0, 10.0),
}


def compute_hours_from_operations(severity: str, operations: List[str]) -> Tuple[int, int]:
    min_total = 0.0
    max_total = 0.0

    for op in operations:
        rng = OPS_HOURS.get(op)
        if rng:
            min_total += float(rng[0])
            max_total += float(rng[1])
        else:
            # Unknown op: add a small buffer instead of breaking
            min_total += 1.0
            max_total += 2.5

    oh = SEVERITY_OVERHEAD.get(severity, (1.0, 3.0))
    min_total += float(oh[0])
    max_total += float(oh[1])

    # Clamp + round to clean integers for UI
    min_i = max(1, int(round(min_total)))
    max_i = max(min_i, int(round(max_total)))
    return min_i, max_i


def risk_note_for(severity: str) -> str:
    if severity == "Severe":
        return "Hidden damage is common in front-corner impacts."
    if severity == "Minor":
        return "Final cost may vary after in-person inspection."
    return "Hidden damage is common in moderate impacts."


def money_fmt(n: int) -> str:
    return f"${n:,}"


# --------------------------------------------------------------------
# AI (optional)
# --------------------------------------------------------------------
def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def analyze_with_ai(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Returns dict with keys:
    severity, confidence, damaged_areas, operations, location, side, summary
    or None on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    try:
        import base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        return None

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are an expert auto body estimator. Analyze the vehicle damage in the photo. "
        "Return STRICT JSON only (no markdown).\n\n"
        "JSON schema:\n"
        "{\n"
        '  "severity": "Minor"|"Moderate"|"Severe",\n'
        '  "confidence": "Low"|"Medium"|"High",\n'
        '  "location": "front"|"rear"|"front-left"|"front-right"|"rear-left"|"rear-right"|"unknown",\n'
        '  "side": "left"|"right"|"unknown",\n'
        '  "damaged_areas": [string, ...],\n'
        '  "operations": [string, ...],\n'
        '  "summary": string\n'
        "}\n\n"
        "Rules:\n"
        "- Use driver POV for left/right.\n"
        "- Operations must be concise and match common shop language like: Replace bumper, Repair fender, Refinish bumper, Replace headlight, Repair hood, Inspect suspension.\n"
        "- If unsure, choose 'unknown' for location/side.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    data = _safe_json_loads(text)
    if not isinstance(data, dict):
        return None

    severity = str(data.get("severity", "")).title()
    if severity not in ("Minor", "Moderate", "Severe"):
        return None

    confidence = str(data.get("confidence", "")).title()
    if confidence not in ("Low", "Medium", "High"):
        confidence = "Medium"

    location = normalize_location(str(data.get("location", "")))
    side = str(data.get("side", "")).lower()
    if side not in ("left", "right", "unknown"):
        side = "unknown"

    damaged_areas = data.get("damaged_areas") if isinstance(data.get("damaged_areas"), list) else []
    operations = data.get("operations") if isinstance(data.get("operations"), list) else []
    summary = str(data.get("summary", "")).strip()

    damaged_areas = [str(x).strip() for x in damaged_areas if str(x).strip()][:10]
    operations = [str(x).strip() for x in operations if str(x).strip()][:12]
    if not summary:
        summary = "Visible damage detected. Further inspection recommended."

    if location == "unknown":
        location = ""
    if side == "unknown":
        side = ""

    return {
        "severity": severity,
        "confidence": confidence,
        "location": location,
        "side": side,
        "damaged_areas": damaged_areas,
        "operations": operations,
        "summary": summary,
    }


# --------------------------------------------------------------------
# Pages (UNCHANGED UI)
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
(function() {
  const form = document.getElementById('estimateForm');
  const btn = document.getElementById('submitBtn');
  const analyzer = document.getElementById('analyzer');
  const fill = document.getElementById('fill');

  function startProgress() {
    let p = 8;
    fill.style.width = p + '%';
    const t = setInterval(() => {
      p = Math.min(96, p + Math.random() * 9);
      fill.style.width = p + '%';
    }, 500);
    return () => {
      clearInterval(t);
      fill.style.width = '100%';
    };
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    btn.disabled = true;
    btn.textContent = 'Analyzing photos...';
    analyzer.style.display = 'block';

    const stop = startProgress();

    try {
      const fd = new FormData(form);
      const r = await fetch('/estimate/api', { method: 'POST', body: fd });
      if (!r.ok) throw new Error('Estimate failed');
      const data = await r.json();
      stop();
      window.location.href = '/estimate/result?id=' + encodeURIComponent(data.estimate_id);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Analyze photo';
      analyzer.style.display = 'none';
      fill.style.width = '0%';
      alert('Something went wrong. Please try again.');
    }
  });
})();
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
    ai_auto: str = Form("1"),  # allow override from form later if needed
):
    k, cfg = resolve_shop(shop_key)

    content = await photo.read()
    filename = photo.filename or ""

    use_ai = AI_AUTO_ON_DEFAULT and (ai_auto.strip() not in ("0", "false", "False", ""))
    ai_out: Optional[Dict[str, Any]] = analyze_with_ai(content) if use_ai else None

    if ai_out:
        severity = ai_out["severity"]
        confidence = ai_out["confidence"]
        damaged_areas = ai_out["damaged_areas"] or []
        operations = ai_out["operations"] or []
        loc = ai_out.get("location", "")
        side = ai_out.get("side", "")
        loc_text = format_location(loc, side)
        summary = ai_out.get("summary", "Visible damage detected. Further inspection recommended.").strip()
        if loc_text:
            summary = summary.rstrip(".") + f" ({loc_text})."
    else:
        severity, confidence = infer_severity(filename, len(content))
        summary, damaged_areas, operations = severity_plan(severity)

        loc, side = infer_location_and_side(filename)
        loc_text = format_location(loc, side)
        if loc_text and "(" not in summary:
            summary = summary.rstrip(".") + f" ({loc_text})."

    # Operation-based hours
    hours_min, hours_max = compute_hours_from_operations(severity, operations)
    risk_note = risk_note_for(severity)

    # Per-shop labor rate applied here
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
