import os
import uuid
import json
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
# NOTE: Keep Miss at 110 so your current $ behavior doesn't shift.
# --------------------------------------------------------------------
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110,  # CAD/hour
    },
}

SHOP_ALIASES: Dict[str, str] = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
    "mississauga_collision_center": "miss",
}

# In-memory store for demo (replace with DB later)
ESTIMATES: Dict[str, Dict[str, Any]] = {}


def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
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
# AI Vision: strict prompt + strict JSON-only contract
# --------------------------------------------------------------------
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")  # change if you want
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


VISION_SYSTEM_PROMPT = (
    "You are an automotive collision damage vision system used by professional body shops.\n\n"
    "Your task is to visually identify damaged vehicle components from photos.\n\n"
    "You must:\n"
    "- Describe ONLY what is clearly visible in the image\n"
    "- Never estimate repair cost\n"
    "- Never estimate labor hours\n"
    "- Never assign severity labels (minor, moderate, severe)\n"
    "- Never give repair advice or recommendations\n"
    "- Never assume hidden damage unless explicitly visible\n"
    "- Never use optimistic language\n\n"
    "Orientation rules:\n"
    "- Left and right are ALWAYS from the DRIVER’S SEATED POV\n"
    '- If unsure about side, choose "unknown"\n\n'
    "Output rules:\n"
    "- You MUST return valid JSON only\n"
    "- Do NOT include explanations\n"
    "- Do NOT include markdown\n"
    "- Do NOT include comments\n"
    "- Do NOT include any text outside the JSON\n"
)

VISION_USER_PROMPT = (
    "Analyze the uploaded vehicle damage photo(s).\n\n"
    "Identify:\n"
    "1. All visibly damaged exterior components\n"
    "2. The damage side using DRIVER POV\n"
    "3. Visible indicators of collision force\n\n"
    "Return the result using the exact JSON schema provided.\n"
    "If a field is unknown, return null.\n"
)

# Allowed values (hard rules)
ALLOWED_PARTS = {
    "bumper",
    "fender",
    "hood",
    "door",
    "headlight",
    "taillight",
    "quarter_panel",
    "wheel",
    "tire",
    "suspension",
    "grille",
    "radiator_support",
    "mirror",
    "unknown",
}
ALLOWED_LOCATION = {
    "front",
    "rear",
    "left",
    "right",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "unknown",
}
ALLOWED_SIDE = {"driver", "passenger", "center", "unknown"}


def _b64_image(upload: UploadFile, content: bytes) -> str:
    # best-effort content-type
    ct = upload.content_type or ""
    if not ct.startswith("image/"):
        # fallback guess from filename
        name = (upload.filename or "").lower()
        if name.endswith(".png"):
            ct = "image/png"
        elif name.endswith(".webp"):
            ct = "image/webp"
        else:
            ct = "image/jpeg"
    return f"data:{ct};base64," + base64.b64encode(content).decode("utf-8")


def _safe_bool(v: Any) -> bool:
    return bool(v) if isinstance(v, (bool, int)) else False


def validate_ai_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal strict validation + normalization
    damaged_parts = obj.get("damaged_parts", [])
    if not isinstance(damaged_parts, list):
        damaged_parts = []

    cleaned_parts = []
    for item in damaged_parts:
        if not isinstance(item, dict):
            continue
        part = (item.get("part") or "unknown").strip().lower()
        location = (item.get("location") or "unknown").strip().lower()
        side = (item.get("side") or "unknown").strip().lower()

        if part not in ALLOWED_PARTS:
            part = "unknown"
        if location not in ALLOWED_LOCATION:
            location = "unknown"
        if side not in ALLOWED_SIDE:
            side = "unknown"

        cleaned_parts.append({"part": part, "location": location, "side": side})

    fi = obj.get("force_indicators") or {}
    if not isinstance(fi, dict):
        fi = {}

    force_indicators = {
        "wheel_displacement": _safe_bool(fi.get("wheel_displacement")),
        "airbag_deployed": _safe_bool(fi.get("airbag_deployed")),
        "debris_field_visible": _safe_bool(fi.get("debris_field_visible")),
        "curb_or_object_strike": _safe_bool(fi.get("curb_or_object_strike")),
        "ride_height_asymmetry": _safe_bool(fi.get("ride_height_asymmetry")),
    }

    conf = obj.get("image_confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))

    return {
        "damaged_parts": cleaned_parts,
        "force_indicators": force_indicators,
        "image_confidence": conf_f,
    }


async def run_vision_ai(photos: List[UploadFile]) -> Optional[Dict[str, Any]]:
    if client is None:
        return None

    # Build multi-image message
    content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": VISION_USER_PROMPT}]

    for p in photos[:3]:
        raw = await p.read()
        if not raw:
            continue
        content_blocks.append(
            {"type": "image_url", "image_url": {"url": _b64_image(p, raw)}}
        )

    # Schema embedded (JSON-only response)
    schema_text = {
        "damaged_parts": [
            {"part": "string", "location": "string", "side": "driver | passenger | center | unknown"}
        ],
        "force_indicators": {
            "wheel_displacement": True,
            "airbag_deployed": False,
            "debris_field_visible": True,
            "curb_or_object_strike": True,
            "ride_height_asymmetry": False,
        },
        "image_confidence": 0.0,
    }
    content_blocks.append(
        {"type": "text", "text": "JSON schema:\n" + json.dumps(schema_text)}
    )

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": content_blocks},
            ],
            # Strongly bias JSON-only output
            response_format={"type": "json_object"},
            temperature=0,
        )
        text = resp.choices[0].message.content or "{}"
        parsed = json.loads(text)
        return validate_ai_json(parsed)
    except Exception:
        return None


# --------------------------------------------------------------------
# Rule Layer (overrides AI optimism) + Operation-based hours
# AI NEVER sets: severity, hours, cost.
# --------------------------------------------------------------------
PART_DISPLAY = {
    "bumper": "Bumper",
    "fender": "Fender",
    "hood": "Hood",
    "door": "Door",
    "headlight": "Headlight",
    "taillight": "Taillight",
    "quarter_panel": "Quarter panel",
    "wheel": "Wheel",
    "tire": "Tire",
    "suspension": "Suspension",
    "grille": "Grille",
    "radiator_support": "Radiator support",
    "mirror": "Mirror",
    "unknown": "Unknown",
}

# Operation hour ranges (min, max)
OP_HOURS: Dict[str, Tuple[int, int]] = {
    "Replace bumper": (4, 7),
    "Repair bumper": (2, 4),
    "Refinish bumper": (2, 4),

    "Replace fender": (5, 9),
    "Repair fender": (3, 6),
    "Refinish fender": (2, 4),

    "Replace headlight": (1, 3),
    "Replace taillight": (1, 2),

    "Repair hood": (4, 8),
    "Replace hood": (6, 10),
    "Refinish hood": (2, 4),

    "Repair door": (4, 8),
    "Replace door": (6, 12),
    "Refinish door": (3, 6),

    "Repair quarter panel": (6, 12),
    "Replace quarter panel": (10, 18),
    "Refinish quarter panel": (3, 6),

    "Inspect suspension": (2, 4),
    "Repair suspension": (6, 12),

    "Inspect wheel/tire": (1, 2),
    "Replace wheel": (1, 2),
    "Replace tire": (1, 2),

    "Replace grille": (1, 3),
    "Replace radiator support": (6, 12),
    "Replace mirror": (1, 2),
}

# Parts that default to replace (conservative) if visible damage is flagged
DEFAULT_REPLACE = {
    "headlight",
    "taillight",
    "mirror",
    "grille",
    "radiator_support",
}


def _side_label(side: str) -> str:
    if side == "driver":
        return "Driver side"
    if side == "passenger":
        return "Passenger side"
    if side == "center":
        return "Center"
    return "Unknown side"


def _loc_label(location: str) -> str:
    # Keep it clean for UI bullets
    mapping = {
        "front": "Front",
        "rear": "Rear",
        "left": "Left",
        "right": "Right",
        "front_left": "Front-left",
        "front_right": "Front-right",
        "rear_left": "Rear-left",
        "rear_right": "Rear-right",
        "unknown": "",
    }
    return mapping.get(location, "")


def build_damaged_areas(ai: Dict[str, Any]) -> List[str]:
    out = []
    for item in ai.get("damaged_parts", []):
        part = item["part"]
        if part == "unknown":
            continue
        loc = _loc_label(item.get("location", "unknown"))
        side = _side_label(item.get("side", "unknown"))
        p = PART_DISPLAY.get(part, part.title())

        # Example: "Fender — Front-right (Driver side)"
        if loc:
            out.append(f"{p} — {loc} ({side})")
        else:
            out.append(f"{p} ({side})")

    # De-dupe while preserving order
    seen = set()
    cleaned = []
    for x in out:
        if x not in seen:
            cleaned.append(x)
            seen.add(x)
    return cleaned or ["Visible damage detected"]


def derive_operations(ai: Dict[str, Any]) -> List[str]:
    parts = [p["part"] for p in ai.get("damaged_parts", []) if p.get("part") in ALLOWED_PARTS]
    fi = ai.get("force_indicators", {}) or {}

    ops: List[str] = []

    # High-force flags
    high_force = (
        fi.get("wheel_displacement")
        or fi.get("airbag_deployed")
        or fi.get("ride_height_asymmetry")
    )

    for part in parts:
        if part == "unknown":
            continue

        # Suspension / wheel logic
        if part == "suspension":
            ops.append("Inspect suspension")
            if high_force:
                ops.append("Repair suspension")
            continue

        if part in ("wheel", "tire"):
            ops.append("Inspect wheel/tire")
            if part == "wheel":
                ops.append("Replace wheel")
            if part == "tire":
                ops.append("Replace tire")
            continue

        # Lights/mirror/grille/rad support conservative replace
        if part in DEFAULT_REPLACE:
            ops.append(f"Replace {PART_DISPLAY[part].lower()}")
            continue

        # Hood logic: repair by default, replace if high-force indicators
        if part == "hood":
            if high_force:
                ops.append("Replace hood")
            else:
                ops.append("Repair hood")
            ops.append("Refinish hood")
            continue

        # Fender / bumper / door / quarter panel: conservative
        if part == "bumper":
            ops.append("Replace bumper")
            ops.append("Refinish bumper")
            continue

        if part == "fender":
            ops.append("Replace fender")
            ops.append("Refinish fender")
            continue

        if part == "door":
            if high_force:
                ops.append("Replace door")
            else:
                ops.append("Repair door")
            ops.append("Refinish door")
            continue

        if part == "quarter_panel":
            if high_force:
                ops.append("Replace quarter panel")
            else:
                ops.append("Repair quarter panel")
            ops.append("Refinish quarter panel")
            continue

    # If AI returned nothing useful, keep a safe generic flow
    ops = [o for o in ops if o in OP_HOURS]

    # De-dupe order
    seen = set()
    cleaned = []
    for o in ops:
        if o not in seen:
            cleaned.append(o)
            seen.add(o)

    return cleaned or ["Inspect damage in person"]


def operation_hours(ops: List[str]) -> Tuple[int, int]:
    hmin = 0
    hmax = 0
    for op in ops:
        rng = OP_HOURS.get(op)
        if not rng:
            continue
        hmin += rng[0]
        hmax += rng[1]
    # Make sure we never show 0–0 if something exists
    if hmin == 0 and hmax == 0 and ops:
        return 3, 6
    return hmin, hmax


def severity_from_rules(ai: Dict[str, Any], ops: List[str]) -> Tuple[str, str, str]:
    """
    Returns (severity, confidence_label, risk_note)
    Rule layer always overrides optimism.
    """
    parts = [p["part"] for p in ai.get("damaged_parts", [])]
    fi = ai.get("force_indicators", {}) or {}
    conf = float(ai.get("image_confidence", 0.0))

    score = 0

    # Parts weight
    weights = {
        "bumper": 2,
        "fender": 3,
        "hood": 3,
        "door": 3,
        "quarter_panel": 4,
        "headlight": 2,
        "taillight": 2,
        "radiator_support": 6,
        "suspension": 7,
        "wheel": 5,
        "tire": 3,
        "grille": 1,
        "mirror": 1,
        "unknown": 0,
    }
    for p in parts:
        score += weights.get(p, 0)

    # Force indicators weight (these override "optimism")
    if fi.get("airbag_deployed"):
        score += 10
    if fi.get("wheel_displacement"):
        score += 9
    if fi.get("ride_height_asymmetry"):
        score += 7
    if fi.get("debris_field_visible"):
        score += 2
    if fi.get("curb_or_object_strike"):
        score += 2

    # Operation-based escalation
    if "Repair suspension" in ops:
        score += 6
    if "Replace radiator support" in ops:
        score += 5
    if "Replace door" in ops or "Replace quarter panel" in ops:
        score += 3

    # Severity thresholds
    if score >= 22:
        severity = "Severe"
        risk = "Hidden damage is common in front-corner impacts."
    elif score >= 10:
        severity = "Moderate"
        risk = "Hidden damage is common in moderate impacts."
    else:
        severity = "Minor"
        risk = "Final cost may vary after in-person inspection."

    # Confidence label (for UI pill only)
    # Note: do not pretend confidence if AI is uncertain.
    if conf >= 0.75:
        confidence_label = "High"
    elif conf >= 0.45:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return severity, confidence_label, risk


def summary_from_parts(severity: str, damaged_areas: List[str], fi: Dict[str, Any]) -> str:
    # Keep this professional, non-hallucinated.
    if severity == "Severe":
        return "Significant visible damage detected. Further inspection is recommended."
    if severity == "Minor":
        return "Light visible damage detected. Further inspection is recommended."
    return "Visible body damage detected. Further inspection recommended."


def money_fmt(n: int) -> str:
    return f"${n:,}"


# --------------------------------------------------------------------
# Fallback heuristic (if AI is unavailable)
# --------------------------------------------------------------------
def infer_severity_fallback(filename: str, size_bytes: int) -> Tuple[str, str]:
    name = (filename or "").lower()
    severe_kw = ["total", "airbag", "frame", "crush", "severe", "structural", "tow"]
    minor_kw = ["scratch", "scuff", "paint", "chip", "minor", "light"]

    if any(k in name for k in severe_kw) or size_bytes > 2_500_000:
        return "Severe", "High"
    if any(k in name for k in minor_kw) or size_bytes < 350_000:
        return "Minor", "High"
    return "Moderate", "Medium"


def severity_plan_fallback(severity: str) -> Tuple[str, List[str], List[str]]:
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

    summary = "Visible body damage detected. Further inspection recommended."
    damaged = ["Bumper", "Fender", "Headlight"]
    ops = ["Replace bumper", "Repair fender", "Replace headlight"]
    return summary, damaged, ops


# --------------------------------------------------------------------
# Pages (keep structure the same; only minimal multi-file capability)
# --------------------------------------------------------------------
def render_landing(shop_key: str, shop_name: str) -> str:
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
      <input type="file" name="photos" accept="image/*" multiple required />
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

      // Cap at 3 photos (quietly)
      const files = fd.getAll('photos');
      fd.delete('photos');
      files.slice(0, 3).forEach(f => fd.append('photos', f));

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
    photos: List[UploadFile] = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)
    photos = photos[:3]  # hard cap

    # Per-shop labor rate
    labor_rate = int(cfg.get("labor_rate", SHOP_CONFIGS["miss"]["labor_rate"]))

    # ------------------------------
    # AI AUTO-ON (Vision) with fallback
    # ------------------------------
    ai = await run_vision_ai(photos)

    if ai is None:
        # Fallback path (never dead-end)
        # Use first photo info only
        first = photos[0]
        raw = await first.read()
        severity, confidence = infer_severity_fallback(first.filename or "", len(raw))
        summary, damaged, ops = severity_plan_fallback(severity)
        ops_clean = [o for o in ops if o in OP_HOURS] or ops
        hmin, hmax = operation_hours(ops_clean)
        risk_note = (
            "Hidden damage is common in front-corner impacts."
            if severity == "Severe"
            else ("Final cost may vary after in-person inspection." if severity == "Minor" else "Hidden damage is common in moderate impacts.")
        )

        cost_min = hmin * labor_rate
        cost_max = hmax * labor_rate

        estimate_id = str(uuid.uuid4())
        ESTIMATES[estimate_id] = {
            "shop_key": k,
            "severity": severity,
            "confidence": confidence,
            "summary": summary,
            "damaged_areas": damaged,
            "operations": ops_clean,
            "labour_hours_min": hmin,
            "labour_hours_max": hmax,
            "cost_min": money_fmt(cost_min),
            "cost_max": money_fmt(cost_max),
            "risk_note": risk_note,
        }
        return JSONResponse({"estimate_id": estimate_id})

    # ------------------------------
    # Rule Layer + Operation-based hours
    # ------------------------------
    damaged_areas = build_damaged_areas(ai)
    operations = derive_operations(ai)
    hmin, hmax = operation_hours(operations)
    severity, confidence_label, risk_note = severity_from_rules(ai, operations)
    summary = summary_from_parts(severity, damaged_areas, ai.get("force_indicators", {}))

    cost_min = hmin * labor_rate
    cost_max = hmax * labor_rate

    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": severity,
        "confidence": confidence_label,
        "summary": summary,
        "damaged_areas": damaged_areas,
        "operations": operations,
        "labour_hours_min": hmin,
        "labour_hours_max": hmax,
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
