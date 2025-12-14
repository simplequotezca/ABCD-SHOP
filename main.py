import os
import re
import json
import base64
import secrets
from datetime import datetime
from typing import List, Dict, Any, Tuple
from html import escape as html_escape

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

app = FastAPI()

# ===============================
# Static files (CSS + logo)
# ===============================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===============================
# Upload storage
# ===============================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Demo-only in-memory session store (token -> data)
SESSIONS: Dict[str, Dict[str, Any]] = {}


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "")).strip("_")
    return (name[:120] or "photo.jpg")


# ===============================
# Shops + Pricing Defaults
# ===============================
DEFAULT_PRICING = {
    "labor_rates": {"body": 95, "paint": 105},  # CAD/hr
    "materials_rate": 38,                       # CAD/hr paint materials
    "contingency_pct": 0.10,                    # 10% buffer for hidden variance
    "range_pct": 0.15,                          # +/- 15% visible range
    "base_floor": {                             # Ontario-ish demo sanity anchors (NOT shown on UI)
        "minor_min": 350, "minor_max": 650,
        "moderate_min": 900, "moderate_max": 1600,
        "severe_min": 2000, "severe_max": 5000
    }
}

# Minimal built-in shop map for demo stability.
# You can later replace this with SHOP_CONFIG_JSON parsing, but do not overcomplicate before you sell.
SHOPS = {
    "mississauga-collision-center": {
        "name": "Mississauga Collision Center",
        # Optional per-shop override via env var JSON:
        # MISS_PRICING_JSON='{"labor_rates":{"body":100,"paint":110},"materials_rate":40,"base_floor":{"severe_min":2300}}'
        "pricing_env_json": "MISS_PRICING_JSON",
    }
}


def get_shop(shop_slug: str) -> Dict[str, Any]:
    if shop_slug in SHOPS:
        shop = dict(SHOPS[shop_slug])
        shop["slug"] = shop_slug
        return shop
    return {"slug": shop_slug, "name": shop_slug.replace("-", " ").title(), "pricing_env_json": ""}


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def get_pricing_for_shop(shop: Dict[str, Any]) -> Dict[str, Any]:
    pricing = deep_copy(DEFAULT_PRICING)
    env_key = shop.get("pricing_env_json") or ""
    if env_key:
        raw = os.getenv(env_key, "").strip()
        if raw:
            try:
                override = json.loads(raw)
                if "labor_rates" in override and isinstance(override["labor_rates"], dict):
                    pricing["labor_rates"].update(override["labor_rates"])
                if "materials_rate" in override:
                    pricing["materials_rate"] = override["materials_rate"]
                if "contingency_pct" in override:
                    pricing["contingency_pct"] = override["contingency_pct"]
                if "range_pct" in override:
                    pricing["range_pct"] = override["range_pct"]
                if "base_floor" in override and isinstance(override["base_floor"], dict):
                    pricing["base_floor"].update(override["base_floor"])
            except Exception:
                # Never crash demo due to bad overrides
                pass
    return pricing


# ===============================
# OpenAI (real photo analysis)
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def normalize_enum(val: str, allowed: List[str], default: str) -> str:
    v = (val or "").strip().title()
    return v if v in allowed else default


def clamp_float(x: Any, default: float = 0.0, lo: float = 0.0, hi: float = 9999.0) -> float:
    try:
        f = float(x)
    except Exception:
        f = default
    return max(lo, min(hi, f))


def analyze_damage_with_openai(image_paths: List[str]) -> Dict[str, Any]:
    """
    Accuracy-first vision analysis with structured output.
    Returns keys:
      severity (Minor/Moderate/Severe)
      confidence (Low/Medium/High)
      damaged_areas (list)
      recommended_ops (list)
      likely_systems_affected (list)  # suspension/steering/structural/ADAS/cooling/etc.
      impact_profile (string)         # short descriptor like "front-right corner"
      labor_hours_body (float)
      labor_hours_paint (float)
      parts_cost (float CAD)
      notes (short)
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in Railway env vars. Add it to enable photo analysis."
        )

    image_parts = []
    for p in image_paths[:3]:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_parts.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    prompt = (
        "You are a senior collision estimator in Ontario. Analyze the vehicle damage from the photos.\n"
        "Be conservative: do NOT claim damage you cannot see. If uncertain, say so.\n"
        "Think like a shop estimator: consider hidden damage likelihood when impacts are near a wheel, suspension, steering, "
        "structural apron/rail areas, or ADAS sensor zones.\n\n"
        "Return ONLY valid JSON with these keys exactly:\n"
        "{\n"
        '  "severity": "Minor|Moderate|Severe",\n'
        '  "confidence": "Low|Medium|High",\n'
        '  "impact_profile": "short descriptor like front-left corner / front-right corner / rear / side",\n'
        '  "damaged_areas": ["..."],\n'
        '  "recommended_ops": ["..."],\n'
        '  "likely_systems_affected": ["suspension","steering","structural","ADAS","cooling","none"],\n'
        '  "labor_hours_body": number,\n'
        '  "labor_hours_paint": number,\n'
        '  "parts_cost": number,\n'
        '  "notes": "1-3 short sentences. Include why replace/repair when possible."\n'
        "}\n\n"
        "Guidelines:\n"
        "- Use realistic labor hours; pick a single reasonable estimate (not a range).\n"
        "- parts_cost is rough CAD for likely replacement parts ONLY if strongly indicated.\n"
        "- If repair vs replace is unclear, lean repair and lower confidence.\n"
        "- If impact is near a wheel/suspension zone, include 'suspension' or 'steering' in likely_systems_affected.\n"
        "- If damage is around headlights/bumper corners on newer cars, consider ADAS calibration risk.\n"
    )

    resp = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}] + image_parts}],
        temperature=0.2
    )
    text = (resp.output_text or "").strip()

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise HTTPException(status_code=500, detail="AI output was not valid JSON. Try again.")

    try:
        data = json.loads(m.group(0))
    except Exception:
        raise HTTPException(status_code=500, detail="AI returned malformed JSON. Try again.")

    data["severity"] = normalize_enum(str(data.get("severity", "")), ["Minor", "Moderate", "Severe"], "Moderate")
    data["confidence"] = normalize_enum(str(data.get("confidence", "")), ["Low", "Medium", "High"], "Medium")
    data["impact_profile"] = str(data.get("impact_profile", "")).strip()[:60] or "Unknown"
    data["notes"] = str(data.get("notes", "")).strip()

    data["labor_hours_body"] = clamp_float(data.get("labor_hours_body", 0), 0.0, 0.0, 80.0)
    data["labor_hours_paint"] = clamp_float(data.get("labor_hours_paint", 0), 0.0, 0.0, 60.0)
    data["parts_cost"] = clamp_float(data.get("parts_cost", 0), 0.0, 0.0, 20000.0)

    if not isinstance(data.get("damaged_areas", []), list):
        data["damaged_areas"] = []
    if not isinstance(data.get("recommended_ops", []), list):
        data["recommended_ops"] = []
    if not isinstance(data.get("likely_systems_affected", []), list):
        data["likely_systems_affected"] = ["none"]

    # normalize systems list
    systems_allowed = {"suspension", "steering", "structural", "adas", "cooling", "none"}
    cleaned = []
    for s in data["likely_systems_affected"]:
        ss = str(s).strip().lower()
        if ss in systems_allowed and ss not in cleaned:
            cleaned.append(ss)
    data["likely_systems_affected"] = cleaned if cleaned else ["none"]

    return data


# ===============================
# Estimation math + severity/risk logic
# ===============================
def money_round10(x: float) -> str:
    return f"${int(round(x / 10) * 10):,}"


def compute_estimate(pricing: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    body_rate = float(pricing["labor_rates"]["body"])
    paint_rate = float(pricing["labor_rates"]["paint"])
    materials_rate = float(pricing["materials_rate"])

    body_hours = max(0.0, float(ai.get("labor_hours_body", 0.0)))
    paint_hours = max(0.0, float(ai.get("labor_hours_paint", 0.0)))
    parts_cost = max(0.0, float(ai.get("parts_cost", 0.0)))

    labor_body = body_hours * body_rate
    labor_paint = paint_hours * paint_rate
    materials = paint_hours * materials_rate

    subtotal = labor_body + labor_paint + materials + parts_cost
    contingency = subtotal * float(pricing.get("contingency_pct", 0.10))
    total = subtotal + contingency

    r = float(pricing.get("range_pct", 0.15))
    low = max(0.0, total * (1 - r))
    high = total * (1 + r))

    return {
        "body_hours": round(body_hours, 1),
        "paint_hours": round(paint_hours, 1),
        "parts_cost": money_round10(parts_cost),
        "labor_body": money_round10(labor_body),
        "labor_paint": money_round10(labor_paint),
        "materials": money_round10(materials),
        "total_mid": total,
        "total_range": f"{money_round10(low)} – {money_round10(high)}",
        "total_mid_str": money_round10(total),
    }


def hidden_damage_risk(ai: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Outputs: (Low/Medium/High, reasons[])
    Uses a mix of AI-provided signals + heuristics on wording.
    """
    reasons = []
    sys = [s.lower() for s in (ai.get("likely_systems_affected") or [])]

    # AI-provided system flags
    if "structural" in sys:
        reasons.append("Possible structural/apron/rail involvement")
    if "suspension" in sys:
        reasons.append("Impact near wheel/suspension zone")
    if "steering" in sys:
        reasons.append("Steering components may be affected")
    if "adas" in sys:
        reasons.append("ADAS/sensor calibration risk")

    # Text heuristics (ops + damaged areas + notes)
    text_blob = " ".join(
        [str(x) for x in (ai.get("damaged_areas") or []) + (ai.get("recommended_ops") or [])] + [ai.get("notes", ""), ai.get("impact_profile", "")]
    ).lower()

    if any(k in text_blob for k in ["wheel", "suspension", "control arm", "tie rod", "knuckle", "strut"]):
        if "Impact near wheel/suspension zone" not in reasons:
            reasons.append("Impact near wheel/suspension zone")
    if any(k in text_blob for k in ["rail", "apron", "frame", "structure", "unibody"]):
        if "Possible structural/apron/rail involvement" not in reasons:
            reasons.append("Possible structural/apron/rail involvement")
    if any(k in text_blob for k in ["radar", "sensor", "adas", "calibration"]):
        if "ADAS/sensor calibration risk" not in reasons:
            reasons.append("ADAS/sensor calibration risk")

    # Score -> risk
    score = 0
    for r in reasons:
        if "structural" in r.lower():
            score += 3
        elif "suspension" in r.lower() or "steering" in r.lower():
            score += 2
        elif "adas" in r.lower():
            score += 1
        else:
            score += 1

    if score >= 4:
        return "High", reasons
    if score >= 2:
        return "Medium", reasons
    return "Low", reasons


def escalate_severity(base: str, risk: str, ai: Dict[str, Any]) -> str:
    """
    Escalates severity one step when hidden risk is high, or certain system flags exist.
    Minor -> Moderate -> Severe
    """
    order = ["Minor", "Moderate", "Severe"]
    if base not in order:
        base = "Moderate"

    sys = [s.lower() for s in (ai.get("likely_systems_affected") or [])]
    force_up = (risk == "High") or ("structural" in sys) or ("suspension" in sys) or ("steering" in sys)

    # Also bump when multiple front-corner components show up together
    blob = " ".join([str(x) for x in (ai.get("damaged_areas") or [])]).lower()
    multi_panel_front = all(k in blob for k in ["bumper", "fender"]) and ("headlight" in blob or "lamp" in blob)

    if force_up or multi_panel_front:
        idx = min(order.index(base) + 1, 2)
        return order[idx]
    return base


def economic_severity(sev: str, est_mid: float, pricing: Dict[str, Any]) -> str:
    """
    Uses pricing base floors to keep severity economically believable.
    If total crosses severe_min, don't label Minor/Moderate.
    """
    bf = pricing.get("base_floor") or {}
    severe_min = float(bf.get("severe_min", 2000))
    moderate_min = float(bf.get("moderate_min", 900))
    minor_max = float(bf.get("minor_max", 650))

    if est_mid >= severe_min:
        return "Severe"
    if est_mid >= moderate_min:
        return "Moderate"
    if est_mid <= minor_max:
        return "Minor"
    return sev


def build_trust_language(risk: str, risk_reasons: List[str]) -> str:
    """
    Standardized phrasing that makes shops trust it and keeps liability low.
    Uses user's requested wording.
    """
    base = "Final repairability and pricing are confirmed after teardown and in-person inspection."
    if risk == "High":
        return (
            "Damage pattern suggests additional hidden damage may be present. "
            "May exceed repair threshold after teardown. " + base
        )
    if risk == "Medium":
        return (
            "Some additional damage may be present that isn’t visible in photos. "
            + base
        )
    return base


# ===============================
# Health check
# ===============================
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ===============================
# Token-gated photo access
# ===============================
@app.get("/u/{token}/{filename}")
def serve_upload(token: str, filename: str):
    sess = SESSIONS.get(token)
    if not sess or filename not in sess.get("filenames", []):
        raise HTTPException(status_code=404, detail="Not found.")

    path = os.path.join(UPLOAD_DIR, token, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found.")

    from fastapi.responses import FileResponse
    return FileResponse(path)


# ===============================
# Landing page
# ===============================
@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>{html_escape(shop_name)} – SimpleQuotez</title>
  <link rel="stylesheet" href="/static/style.css?v=ESTV1">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>{html_escape(shop_name)}</h1>
    <p class="subtitle">Upload photos to get a fast AI repair estimate.</p>

    <a class="cta" href="/quote/{html_escape(shop_slug)}/upload">Start Estimate</a>

    <div class="upload-hint">
      <strong>Best results with 3 photos:</strong>
      <ul>
        <li>Overall damage</li>
        <li>Close-up</li>
        <li>Side angle</li>
      </ul>
    </div>

    <div class="note">Preliminary estimate · Final pricing after inspection</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ===============================
# Upload page
# ===============================
@app.get("/quote/{shop_slug}/upload", response_class=HTMLResponse)
def upload_page(shop_slug: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>Upload Photos – {html_escape(shop_name)}</title>
  <link rel="stylesheet" href="/static/style.css?v=ESTV1">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>Upload Photos</h1>
    <p class="subtitle">Add 1–3 photos of the damage.</p>

    <form class="form" action="/quote/{html_escape(shop_slug)}/upload" method="post" enctype="multipart/form-data">
      <input class="file" type="file" name="photos" accept="image/*" multiple required />
      <button type="submit">Analyze Damage</button>
    </form>

    <div class="note">Tip: Overall shot + close-up = best results.</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ===============================
# Upload handler -> runs OpenAI analysis -> redirects to result page
# ===============================
@app.post("/quote/{shop_slug}/upload")
async def upload_post(shop_slug: str, photos: List[UploadFile] = File(...)):
    if not photos:
        raise HTTPException(status_code=400, detail="Upload at least 1 photo.")
    if len(photos) > 3:
        photos = photos[:3]

    token = secrets.token_urlsafe(18)
    folder = os.path.join(UPLOAD_DIR, token)
    os.makedirs(folder, exist_ok=True)

    filenames = []
    paths = []

    for i, up in enumerate(photos):
        fn = safe_filename(up.filename or f"photo_{i+1}.jpg")
        path = os.path.join(folder, fn)
        content = await up.read()
        with open(path, "wb") as f:
            f.write(content)
        filenames.append(fn)
        paths.append(path)

    # Store session early so photo links work even if AI fails.
    SESSIONS[token] = {
        "shop_slug": shop_slug,
        "filenames": filenames,
        "created_at": datetime.utcnow().isoformat(),
    }

    ai = analyze_damage_with_openai(paths)

    # Pricing + estimate
    shop = get_shop(shop_slug)
    pricing = get_pricing_for_shop(shop)
    est = compute_estimate(pricing, ai)

    # Risk + severity calibration
    risk, reasons = hidden_damage_risk(ai)
    sev1 = escalate_severity(ai.get("severity", "Moderate"), risk, ai)
    sev2 = economic_severity(sev1, est["total_mid"], pricing)

    # Lock the calibrated values into session
    ai["severity_raw"] = ai.get("severity", "Moderate")
    ai["severity"] = sev2
    ai["hidden_damage_risk"] = risk
    ai["risk_reasons"] = reasons

    # Trust language + price framing
    trust = build_trust_language(risk, reasons)
    price_framing = "Estimate reflects typical repair pricing for comparable collision damage. Final cost varies by shop labor rates, parts availability, and findings after teardown."

    ai["trust_language"] = trust
    ai["price_framing"] = price_framing

    SESSIONS[token]["ai"] = ai
    SESSIONS[token]["est"] = est

    return RedirectResponse(url=f"/quote/{shop_slug}/result/{token}", status_code=303)


# ===============================
# AI Estimate Results Page
# ===============================
@app.get("/quote/{shop_slug}/result/{token}", response_class=HTMLResponse)
def result_page(shop_slug: str, token: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]

    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=404, detail="Session expired.")

    ai = sess.get("ai")
    est = sess.get("est")
    if not ai or not est:
        raise HTTPException(status_code=500, detail="Missing analysis/estimate for this session.")

    # Photo links
    photo_links = "".join(
        f'<a class="photo-link" href="/u/{html_escape(token)}/{html_escape(fn)}" target="_blank" rel="noopener">View photo {i+1}</a>'
        for i, fn in enumerate(sess.get("filenames", []))
    )

    damaged_areas = ai.get("damaged_areas", [])[:10]
    recommended_ops = ai.get("recommended_ops", [])[:12]

    # FIX: use html_escape instead of re.escape (which caused backslashes)
    damaged_html = "".join(f"<li>{html_escape(str(x))}</li>" for x in damaged_areas) or "<li>Not enough visibility to confirm specific panels.</li>"
    ops_html = "".join(f"<li>{html_escape(str(x))}</li>" for x in recommended_ops) or "<li>Repair/refinish likely — inspection required to confirm.</li>"

    severity = html_escape(ai.get("severity", "Moderate"))
    confidence = html_escape(ai.get("confidence", "Medium"))
    risk = html_escape(ai.get("hidden_damage_risk", "Medium"))
    impact = html_escape(ai.get("impact_profile", "Unknown"))

    summary = ai.get("notes", "").strip()
    if not summary:
        summary = "Inspection recommended to confirm hidden damage and repair strategy."

    trust_line = ai.get("trust_language", "")
    price_framing = ai.get("price_framing", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>AI Estimate – {html_escape(shop_name)}</title>
  <link rel="stylesheet" href="/static/style.css?v=ESTV1">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>AI Estimate</h1>

    <div class="pill">{severity} · {confidence} confidence</div>
    <div class="pill pill2">Hidden damage risk: {risk} · {impact}</div>

    <div class="block">
      <div class="label">Summary</div>
      <div class="text">{html_escape(summary)}</div>
      <div class="mini">{html_escape(trust_line)}</div>
    </div>

    <div class="block">
      <div class="label">Likely damaged areas</div>
      <ul class="list">{damaged_html}</ul>
    </div>

    <div class="block">
      <div class="label">Recommended operations</div>
      <ul class="list">{ops_html}</ul>
    </div>

    <div class="block">
      <div class="label">Estimate (CAD)</div>
      <div class="text big">{html_escape(est["total_range"])}</div>
      <div class="mini">
        Body: {est["body_hours"]}h ({html_escape(est["labor_body"])}) ·
        Paint: {est["paint_hours"]}h ({html_escape(est["labor_paint"])})<br/>
        Materials: {html_escape(est["materials"])} · Parts (rough): {html_escape(est["parts_cost"])}<br/>
        {html_escape(price_framing)}
      </div>
    </div>

    <div class="block">
      <div class="label">Photos</div>
      <div class="photos">{photo_links}</div>
    </div>

    <a class="cta" href="/quote/{html_escape(shop_slug)}">Back to start</a>

    <div class="note">Preliminary estimate. Final pricing after in-person inspection.</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
