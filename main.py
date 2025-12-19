import os
import uuid
import json
import base64
from typing import Dict, Any, Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# OpenAI (requirements: openai==1.30.5)
from openai import OpenAI

app = FastAPI()

# Serve static files (CSS + logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")  # change via env if needed

# AI influence tuning (LABOR ONLY)
# 0.00 = rules-only, 0.30 = recommended, 0.50+ = aggressive
AI_LABOR_INFLUENCE = float(os.getenv("AI_LABOR_INFLUENCE", "0.30"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Shop config (add more shops here)
# NOTE: Keep Miss at 110 so your current $ ranges don't swing wildly.
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {
        "name": "Mississauga Collision Center",
        "labor_rate": 110,  # CAD/hour
    },
}

# Aliases / slugs that should resolve to canonical keys
SHOP_ALIASES: Dict[str, str] = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
    "mississauga_collision_center": "miss",
}

# In-memory store for demo
ESTIMATES: Dict[str, Dict[str, Any]] = {}


def resolve_shop(shop_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    key_raw = (shop_key or "miss").strip()
    canonical = SHOP_ALIASES.get(key_raw, key_raw)

    cfg = SHOP_CONFIGS.get(canonical)
    if cfg:
        return canonical, cfg

    fallback = {
        "name": key_raw.replace("-", " ").title(),
        "labor_rate": SHOP_CONFIGS["miss"]["labor_rate"],
    }
    return canonical, fallback


def money_fmt(n: int) -> str:
    return f"${n:,}"


# -----------------------------
# AI VISION JSON CONTRACT (STRICT)
# -----------------------------
AI_VISION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "ai_bodyshop_estimate",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string", "enum": ["Minor", "Moderate", "Severe"]},
                    "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
                    "impact_zones": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Front",
                                "Rear",
                                "DriverSide",
                                "PassengerSide",
                                "FrontDriverCorner",
                                "FrontPassengerCorner",
                                "RearDriverCorner",
                                "RearPassengerCorner",
                                "Unknown",
                            ],
                        },
                        "minItems": 0,
                        "maxItems": 6,
                    },
                    "notes": {"type": "string"},
                    "structural_suspected": {"type": "boolean"},
                    "mechanical_suspected": {"type": "boolean"},
                    "airbag_suspected": {"type": "boolean"},
                },
                "required": [
                    "severity",
                    "confidence",
                    "impact_zones",
                    "notes",
                    "structural_suspected",
                    "mechanical_suspected",
                    "airbag_suspected",
                ],
            },
            "damage_items": {
                "type": "array",
                "minItems": 0,
                "maxItems": 18,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "component": {
                            "type": "string",
                            "enum": [
                                "Front bumper",
                                "Rear bumper",
                                "Bumper reinforcement",
                                "Grille",
                                "Radiator support",
                                "Hood",
                                "Fender",
                                "Door",
                                "Quarter panel",
                                "Headlight",
                                "Taillight",
                                "Mirror",
                                "Wheel",
                                "Tire",
                                "Suspension",
                                "Cooling system",
                                "Frame/Unibody",
                                "Windshield/Glass",
                                "Unknown",
                            ],
                        },
                        "position": {"type": "string", "enum": ["Front", "Rear", "Side", "Unknown"]},
                        # IMPORTANT: must be DRIVER POV
                        "side_driver_pov": {
                            "type": "string",
                            "enum": ["Driver", "Passenger", "Center", "Unknown"],
                        },
                        "damage_type": {
                            "type": "string",
                            "enum": [
                                "Scratch",
                                "Dent",
                                "Crack",
                                "Tear",
                                "Broken",
                                "Missing",
                                "Misaligned",
                                "Deformed",
                                "Unknown",
                            ],
                        },
                        "action": {"type": "string", "enum": ["Inspect", "Repair", "Replace", "Refinish"]},
                        "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
                    },
                    "required": [
                        "component",
                        "position",
                        "side_driver_pov",
                        "damage_type",
                        "action",
                        "confidence",
                    ],
                },
            },
            "recommended_ops": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {"type": "string"},
            },
        },
        "required": ["overall", "damage_items", "recommended_ops"],
    },
}


AI_VISION_SYSTEM_PROMPT = """You are an expert collision estimator for a body shop.
You MUST:
- Interpret left/right strictly from DRIVER'S point of view.
- Be conservative: if uncertain, choose Inspect + lower confidence, and widen repair scope.
- If visible impact is in a front corner, include typical related components (bumper/fender/headlight) if plausibly affected.
- If you see wheel angle issues, suspension deformation, major gaps/misalignment, or heavy corner intrusion, flag mechanical_suspected or structural_suspected.
- Output ONLY valid JSON matching the provided schema. No extra keys, no markdown."""

# -----------------------------
# OPERATION-BASED HOURS (RULE LAYER)
# Conservative ranges; can be tuned later.
# -----------------------------
OP_HOURS: Dict[str, Tuple[int, int]] = {
    # Bumper
    "Replace front bumper": (4, 6),
    "Replace rear bumper": (4, 6),
    "Refinish bumper": (3, 4),
    "Repair bumper": (2, 4),

    # Fender / hood / panels
    "Replace fender": (4, 6),
    "Repair fender": (3, 5),
    "Refinish fender": (2, 3),
    "Repair hood": (3, 6),
    "Replace hood": (4, 7),
    "Refinish hood": (3, 4),

    # Lights
    "Replace headlight": (1, 2),
    "Replace taillight": (1, 2),

    # Structural / mechanical flags
    "Structural inspection": (3, 6),
    "Suspension inspection": (2, 4),
    "Alignment check": (1, 2),

    # Common
    "Scan for codes (pre/post)": (1, 2),
    "Test drive / verify": (1, 2),
}


def normalize_op(op: str) -> str:
    """Normalize AI ops into our op keys."""
    o = op.strip()

    # Keep it strict and predictable (small mapping only)
    mapping = {
        "Replace bumper": "Replace front bumper",
        "Replace front bumper": "Replace front bumper",
        "Replace rear bumper": "Replace rear bumper",
        "Repair bumper": "Repair bumper",
        "Refinish bumper": "Refinish bumper",

        "Replace fender": "Replace fender",
        "Repair fender": "Repair fender",
        "Refinish fender": "Refinish fender",

        "Replace headlight": "Replace headlight",
        "Replace taillight": "Replace taillight",

        "Repair hood": "Repair hood",
        "Replace hood": "Replace hood",
        "Refinish hood": "Refinish hood",

        "Inspect suspension": "Suspension inspection",
        "Suspension inspection": "Suspension inspection",
        "Inspect structural": "Structural inspection",
        "Structural inspection": "Structural inspection",

        "Alignment": "Alignment check",
        "Alignment check": "Alignment check",

        "Scan": "Scan for codes (pre/post)",
        "Scan for codes": "Scan for codes (pre/post)",
        "Scan for codes (pre/post)": "Scan for codes (pre/post)",

        "Test drive": "Test drive / verify",
        "Test drive / verify": "Test drive / verify",
    }
    return mapping.get(o, o)


def compute_hours_from_ops(ops: List[str]) -> Tuple[int, int]:
    """
    Sum min/max hours for known ops. Unknown ops add conservative padding.
    """
    seen = []
    for op in ops:
        n = normalize_op(op)
        if n not in seen:
            seen.append(n)

    hmin = 0
    hmax = 0
    unknown_count = 0

    for op in seen:
        if op in OP_HOURS:
            a, b = OP_HOURS[op]
            hmin += a
            hmax += b
        else:
            unknown_count += 1

    # Unknown operations: add conservative buffer
    if unknown_count:
        hmin += 1 * unknown_count
        hmax += 3 * unknown_count

    # Clamp to sane bounds
    hmin = max(1, hmin)
    hmax = max(hmin + 1, hmax)
    return hmin, hmax


# -----------------------------
# AI LABOR INFLUENCE (LABOR ONLY)
# -----------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def confidence_multiplier(confidence: str) -> float:
    # Conservative tuning:
    # High -> slightly higher hours (more certainty that work is real)
    # Medium -> neutral
    # Low -> slightly lower midpoint (but rules still clamp floors/ceilings)
    c = (confidence or "").strip().lower()
    if c == "high":
        return 1.12
    if c == "medium":
        return 1.00
    if c == "low":
        return 0.90
    return 1.00


def ai_adjusted_labor_range(rule_min: int, rule_max: int, confidence: str) -> Tuple[float, float]:
    """
    Build an AI-influenced labor range by shifting the midpoint based on confidence,
    then clamping it within the rule bounds.
    """
    rule_min_f = float(rule_min)
    rule_max_f = float(rule_max)

    base_mid = (rule_min_f + rule_max_f) / 2.0
    half_span = (rule_max_f - rule_min_f) / 2.0

    mult = confidence_multiplier(confidence)
    ai_mid = base_mid * mult

    ai_min = ai_mid - half_span
    ai_max = ai_mid + half_span

    # Clamp AI range into rule bounds so AI can only move *within* safe rails
    ai_min = _clamp(ai_min, rule_min_f, rule_max_f)
    ai_max = _clamp(ai_max, ai_min, rule_max_f)

    # Ensure at least 1 hour spread
    if ai_max - ai_min < 1.0:
        ai_max = _clamp(ai_min + 1.0, ai_min, rule_max_f)

    return ai_min, ai_max


def blend_labor_ranges(rule_min: int, rule_max: int, ai_min: float, ai_max: float) -> Tuple[int, int]:
    """
    Blend AI and rules at labor level ONLY, then clamp to the rule bounds.
    """
    influence = _clamp(float(AI_LABOR_INFLUENCE), 0.0, 1.0)

    rmin = float(rule_min)
    rmax = float(rule_max)

    final_min = (rmin * (1.0 - influence)) + (ai_min * influence)
    final_max = (rmax * (1.0 - influence)) + (ai_max * influence)

    # Final hard clamp to rule rails
    final_min = _clamp(final_min, rmin, rmax)
    final_max = _clamp(final_max, final_min, rmax)

    # Round + ensure spread
    fmin_i = int(round(final_min))
    fmax_i = int(round(final_max))
    if fmax_i <= fmin_i:
        fmax_i = min(rule_max, fmin_i + 1)

    # Safety: keep inside rule bounds
    fmin_i = max(rule_min, min(fmin_i, rule_max))
    fmax_i = max(fmin_i + 1, min(fmax_i, rule_max))

    return fmin_i, fmax_i


# -----------------------------
# RULE OVERRIDES (CLAMP OPTIMISM)
# -----------------------------
def apply_rule_overrides(ai: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overrides to prevent optimistic / unrealistic results.
    """
    overall = ai.get("overall", {})
    severity = overall.get("severity", "Moderate")
    confidence = overall.get("confidence", "Medium")

    structural = bool(overall.get("structural_suspected", False))
    mechanical = bool(overall.get("mechanical_suspected", False))
    airbag = bool(overall.get("airbag_suspected", False))

    # If any of these flags: severity cannot be Minor
    if structural or mechanical or airbag:
        severity = "Severe"

    # If confidence is Low: force conservative adds
    ops = list(ai.get("recommended_ops", []) or [])
    damage_items = list(ai.get("damage_items", []) or [])

    # Always include scan (body shops actually do this, and it helps realism)
    if "Scan for codes (pre/post)" not in ops:
        ops.append("Scan for codes (pre/post)")

    # If structural suspected: add structural inspection
    if structural and "Structural inspection" not in ops:
        ops.append("Structural inspection")

    # If mechanical suspected: add suspension inspection + alignment
    if mechanical:
        if "Suspension inspection" not in ops:
            ops.append("Suspension inspection")
        if "Alignment check" not in ops:
            ops.append("Alignment check")

    # Low confidence -> add inspect and widen range (hours computed later)
    if confidence == "Low":
        if "Test drive / verify" not in ops:
            ops.append("Test drive / verify")

    # Ensure driver/passenger wording is present in damage items (driver POV)
    # If unknown side, keep Unknown (don’t hallucinate).
    cleaned_damage = []
    for d in damage_items:
        side = d.get("side_driver_pov", "Unknown")
        comp = d.get("component", "Unknown")
        cleaned_damage.append({
            "component": comp,
            "position": d.get("position", "Unknown"),
            "side_driver_pov": side,
            "damage_type": d.get("damage_type", "Unknown"),
            "action": d.get("action", "Inspect"),
            "confidence": d.get("confidence", "Medium"),
        })

    ai["overall"]["severity"] = severity
    ai["overall"]["confidence"] = confidence
    ai["damage_items"] = cleaned_damage

    # Normalize ops
    norm_ops = []
    for o in ops:
        n = normalize_op(o)
        if n not in norm_ops:
            norm_ops.append(n)
    ai["recommended_ops"] = norm_ops

    return ai


def build_summary(overall: Dict[str, Any]) -> str:
    sev = overall.get("severity", "Moderate")
    structural = overall.get("structural_suspected", False)
    mechanical = overall.get("mechanical_suspected", False)

    if sev == "Severe":
        if structural:
            return "Significant damage with possible structural involvement."
        if mechanical:
            return "Significant damage with possible mechanical involvement."
        return "Significant damage detected. Further inspection recommended."

    if sev == "Minor":
        return "Light cosmetic damage likely. Further inspection recommended."

    return "Visible body damage detected. Further inspection recommended."


def risk_note_for(severity: str) -> str:
    if severity == "Severe":
        return "Hidden damage is common in front-corner impacts."
    if severity == "Minor":
        return "Final cost may vary after in-person inspection."
    return "Hidden damage is common in moderate impacts."


def bullets(items: List[str]) -> str:
    return "\n".join(f"<li>{i}</li>" for i in items)


def format_damage_bullets(damage_items: List[Dict[str, Any]]) -> List[str]:
    out = []
    for d in damage_items:
        comp = d.get("component", "Unknown")
        side = d.get("side_driver_pov", "Unknown")
        if side in ["Driver", "Passenger"]:
            out.append(f"{side} {comp}".replace("  ", " ").strip())
        else:
            out.append(f"{comp}".strip())
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen[:10]


# -----------------------------
# AI CALL (AUTO-ON)
# -----------------------------
async def ai_vision_analyze(photos: List[UploadFile]) -> Dict[str, Any]:
    """
    Uses AI vision to produce JSON matching schema.
    If AI is unavailable or fails, returns a conservative fallback.
    """
    files = photos[:3]
    images_payload = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        b64 = base64.b64encode(data).decode("utf-8")
        images_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    if not client or not images_payload:
        return {
            "overall": {
                "severity": "Moderate",
                "confidence": "Low",
                "impact_zones": ["Unknown"],
                "notes": "AI vision unavailable; using conservative fallback.",
                "structural_suspected": False,
                "mechanical_suspected": False,
                "airbag_suspected": False,
            },
            "damage_items": [
                {
                    "component": "Front bumper",
                    "position": "Front",
                    "side_driver_pov": "Unknown",
                    "damage_type": "Unknown",
                    "action": "Inspect",
                    "confidence": "Low",
                }
            ],
            "recommended_ops": ["Scan for codes (pre/post)", "Structural inspection"],
        }

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": AI_VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the vehicle damage from these photos and return JSON only."},
                        *images_payload,
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": AI_VISION_JSON_SCHEMA},
            temperature=0.2,
        )

        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)

        if not isinstance(data, dict) or "overall" not in data:
            raise ValueError("Invalid AI JSON")

        return data

    except Exception:
        return {
            "overall": {
                "severity": "Moderate",
                "confidence": "Low",
                "impact_zones": ["Unknown"],
                "notes": "AI analysis failed; using conservative fallback.",
                "structural_suspected": False,
                "mechanical_suspected": False,
                "airbag_suspected": False,
            },
            "damage_items": [
                {
                    "component": "Front bumper",
                    "position": "Front",
                    "side_driver_pov": "Unknown",
                    "damage_type": "Unknown",
                    "action": "Inspect",
                    "confidence": "Low",
                }
            ],
            "recommended_ops": ["Scan for codes (pre/post)", "Structural inspection"],
        }


# -----------------------------
# HTML PAGES (MATCH YOUR CSS CLASSES)
# -----------------------------
def render_landing(shop_key: str, shop_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
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
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
  <title>{shop_name} — Upload</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="page">
  <div class="card">
    <div class="title">{shop_name}</div>
    <div class="subtitle">Upload 1–3 photos for an AI repair estimate.</div>

    <form id="estimateForm">
      <input type="hidden" name="shop_key" value="{shop_key}" />
      <input id="photoInput" type="file" name="photos" accept="image/*" multiple required />
      <button class="cta" id="submitBtn" type="submit">Analyze</button>
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
  const input = document.getElementById('photoInput');

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

    const files = input.files ? Array.from(input.files) : [];
    if (files.length < 1) {{
      alert('Please choose at least 1 photo.');
      return;
    }}
    if (files.length > 3) {{
      alert('Please choose up to 3 photos.');
      return;
    }}

    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    analyzer.style.display = 'block';

    const stop = startProgress();

    try {{
      const fd = new FormData();
      fd.append('shop_key', '{shop_key}');
      files.forEach(f => fd.append('photos', f));

      const r = await fetch('/estimate/api', {{ method: 'POST', body: fd }});
      if (!r.ok) throw new Error('Estimate failed');
      const data = await r.json();
      stop();
      window.location.href = '/estimate/result?id=' + encodeURIComponent(data.estimate_id);
    }} catch (err) {{
      btn.disabled = false;
      btn.textContent = 'Analyze';
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


def render_result(data: Dict[str, Any]) -> str:
    pill = f"{data['severity']} • {data['confidence']} confidence"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
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


# -----------------------------
# ROUTES
# -----------------------------
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

    # Hard cap to 3 (even if client sends more)
    photos = (photos or [])[:3]

    # 1) AI Vision (auto-on)
    ai_json = await ai_vision_analyze(photos)

    # 2) Rule overrides (clamp optimism)
    ai_json = apply_rule_overrides(ai_json)

    overall = ai_json["overall"]
    severity = overall["severity"]
    confidence = overall["confidence"]

    # Damage bullets + ops
    damaged_areas = format_damage_bullets(ai_json.get("damage_items", []))
    operations = [normalize_op(o) for o in (ai_json.get("recommended_ops", []) or [])]

    # 3) Operation-based hours (RULE RANGE)
    hours_min, hours_max = compute_hours_from_ops(operations)

    # Final severity clamp based on hours (prevents "Severe" with 4 hours nonsense)
    if severity == "Severe" and hours_min < 18:
        hours_min = 18
        hours_max = max(hours_max, 28)

    # If confidence low -> widen range a bit (existing behavior preserved)
    if confidence == "Low":
        hours_min = max(6, hours_min)
        hours_max = max(hours_max, int(hours_max * 1.25))

    # 4) AI influence tuning (LABOR ONLY) — blended INSIDE rule rails
    ai_min_f, ai_max_f = ai_adjusted_labor_range(hours_min, hours_max, confidence)
    hours_min, hours_max = blend_labor_ranges(hours_min, hours_max, ai_min_f, ai_max_f)

    labor_rate = int(cfg.get("labor_rate", SHOP_CONFIGS["miss"]["labor_rate"]))
    cost_min = hours_min * labor_rate
    cost_max = hours_max * labor_rate

    summary = build_summary(overall)
    risk_note = risk_note_for(severity)

    estimate_id = str(uuid.uuid4())
    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "damaged_areas": damaged_areas[:10],
        "operations": operations[:12],
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
