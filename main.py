import os
import re
import json
import base64
import secrets
from datetime import datetime
from typing import List, Dict, Any

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
    "labor_rates": {"body": 95, "paint": 105},   # CAD/hr (editable per shop later)
    "materials_rate": 38,                         # CAD/hr paint materials
    "contingency_pct": 0.10,                      # 10% buffer for hidden damage variance
    "range_pct": 0.15                             # show +/- 15% range around computed total
}

SHOPS = {
    "mississauga-collision-center": {
        "name": "Mississauga Collision Center",
        # optional per-shop override via env var containing JSON:
        # MISS_PRICING_JSON='{"labor_rates":{"body":100,"paint":110},"materials_rate":40}'
        "pricing_env_json": "MISS_PRICING_JSON",
    }
}

def get_shop(shop_slug: str) -> Dict[str, Any]:
    if shop_slug in SHOPS:
        shop = dict(SHOPS[shop_slug])
        shop["slug"] = shop_slug
        return shop
    return {"slug": shop_slug, "name": shop_slug.replace("-", " ").title(), "pricing_env_json": ""}

def get_pricing_for_shop(shop: Dict[str, Any]) -> Dict[str, Any]:
    pricing = json.loads(json.dumps(DEFAULT_PRICING))  # deep copy
    env_key = shop.get("pricing_env_json") or ""
    if env_key:
        raw = os.getenv(env_key, "").strip()
        if raw:
            try:
                override = json.loads(raw)
                # shallow merge for expected keys
                if "labor_rates" in override and isinstance(override["labor_rates"], dict):
                    pricing["labor_rates"].update(override["labor_rates"])
                if "materials_rate" in override:
                    pricing["materials_rate"] = override["materials_rate"]
                if "contingency_pct" in override:
                    pricing["contingency_pct"] = override["contingency_pct"]
                if "range_pct" in override:
                    pricing["range_pct"] = override["range_pct"]
            except Exception:
                # If override JSON is bad, ignore it (demo must not crash)
                pass
    return pricing

# ===============================
# OpenAI (real photo analysis)
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def analyze_damage_with_openai(image_paths: List[str]) -> Dict[str, Any]:
    """
    Uses OpenAI vision to infer:
      - severity (Minor/Moderate/Severe)
      - damaged_areas (list)
      - recommended_ops (list)
      - labor_hours_body (float)
      - labor_hours_paint (float)
      - parts_cost (int CAD)
      - confidence (Low/Medium/High)
      - notes (string)
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in Railway env vars. Add it to enable photo analysis."
        )

    # attach up to 3 images (your flow already caps at 3)
    image_parts = []
    for p in image_paths[:3]:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_parts.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    # Accuracy-first prompt: conservative + structured + avoids hallucinations
    prompt = (
        "You are a senior collision estimator in Ontario. Analyze the vehicle damage from the photos.\n"
        "Be conservative: do not claim damage you cannot see. If uncertain, say so.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "{\n"
        '  "severity": "Minor|Moderate|Severe",\n'
        '  "damaged_areas": ["..."],\n'
        '  "recommended_ops": ["..."],\n'
        '  "labor_hours_body": number,\n'
        '  "labor_hours_paint": number,\n'
        '  "parts_cost": number,\n'
        '  "confidence": "Low|Medium|High",\n'
        '  "notes": "short, practical notes (1-3 sentences)"\n'
        "}\n\n"
        "Guidelines:\n"
        "- labor hours should be realistic ranges converted to a single estimate\n"
        "- parts_cost is a rough CAD placeholder if a panel/bumper/light likely needs replacement\n"
        "- if repair vs replace is unclear, lean to repair and lower confidence\n"
    )

    resp = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}] + image_parts
        }],
        temperature=0.2
    )

    text = (resp.output_text or "").strip()

    # Extract JSON robustly
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise HTTPException(status_code=500, detail="AI output was not valid JSON. Try again.")
    try:
        data = json.loads(m.group(0))
    except Exception:
        raise HTTPException(status_code=500, detail="AI returned malformed JSON. Try again.")

    # minimal normalization
    data["severity"] = str(data.get("severity", "Moderate")).title()
    if data["severity"] not in ("Minor", "Moderate", "Severe"):
        data["severity"] = "Moderate"

    for k in ["labor_hours_body", "labor_hours_paint", "parts_cost"]:
        try:
            data[k] = float(data.get(k, 0)) if k != "parts_cost" else float(data.get(k, 0))
        except Exception:
            data[k] = 0.0

    if not isinstance(data.get("damaged_areas", []), list):
        data["damaged_areas"] = []
    if not isinstance(data.get("recommended_ops", []), list):
        data["recommended_ops"] = []

    data["confidence"] = str(data.get("confidence", "Medium")).title()
    if data["confidence"] not in ("Low", "Medium", "High"):
        data["confidence"] = "Medium"

    data["notes"] = str(data.get("notes", "")).strip()

    return data

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
    low = max(0, total * (1 - r))
    high = total * (1 + r)

    def money(x: float) -> str:
        return f"${int(round(x / 10) * 10):,}"

    return {
        "body_hours": round(body_hours, 1),
        "paint_hours": round(paint_hours, 1),
        "parts_cost": money(parts_cost),
        "labor_body": money(labor_body),
        "labor_paint": money(labor_paint),
        "materials": money(materials),
        "total_range": f"{money(low)} – {money(high)}",
        "total_mid": money(total),
    }

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
  <title>{shop_name} – SimpleQuotez</title>
  <link rel="stylesheet" href="/static/style.css?v=STEP3">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>{shop_name}</h1>
    <p class="subtitle">Upload photos to get a fast AI repair estimate.</p>

    <a class="cta" href="/quote/{shop_slug}/upload">Start Estimate</a>

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
  <title>Upload Photos – {shop_name}</title>
  <link rel="stylesheet" href="/static/style.css?v=STEP3">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>Upload Photos</h1>
    <p class="subtitle">Add 1–3 photos of the damage.</p>

    <form class="form" action="/quote/{shop_slug}/upload" method="post" enctype="multipart/form-data">
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

    # Store basic session first (so photo links work even if AI fails)
    SESSIONS[token] = {
        "shop_slug": shop_slug,
        "filenames": filenames,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Run OpenAI analysis (required for your goal)
    ai = analyze_damage_with_openai(paths)

    # Persist analysis
    SESSIONS[token]["ai"] = ai

    return RedirectResponse(url=f"/quote/{shop_slug}/result/{token}", status_code=303)

# ===============================
# AI Estimate Results Page (Step 3)
# ===============================
@app.get("/quote/{shop_slug}/result/{token}", response_class=HTMLResponse)
def result_page(shop_slug: str, token: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]
    pricing = get_pricing_for_shop(shop)

    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=404, detail="Session expired.")

    ai = sess.get("ai")
    if not ai:
        raise HTTPException(status_code=500, detail="No AI analysis found for this session.")

    est = compute_estimate(pricing, ai)

    photo_links = "".join(
        f'<a class="photo-link" href="/u/{token}/{fn}" target="_blank" rel="noopener">View photo {i+1}</a>'
        for i, fn in enumerate(sess["filenames"])
    )

    damaged_areas = ai.get("damaged_areas", [])
    recommended_ops = ai.get("recommended_ops", [])

    damaged_html = "".join(f"<li>{re.escape(str(x))}</li>" for x in damaged_areas[:8]) or "<li>Not enough visibility to confirm specific panels.</li>"
    ops_html = "".join(f"<li>{re.escape(str(x))}</li>" for x in recommended_ops[:10]) or "<li>Repair/refinish likely — inspection required to confirm.</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>AI Estimate – {shop_name}</title>
  <link rel="stylesheet" href="/static/style.css?v=STEP3">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>AI Estimate</h1>

    <div class="pill">{ai.get("severity","Moderate")} · {ai.get("confidence","Medium")} confidence</div>

    <div class="block">
      <div class="label">Summary</div>
      <div class="text">{ai.get("notes","").strip() or "Inspection recommended to confirm hidden damage and repair strategy."}</div>
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
      <div class="text big">{est["total_range"]}</div>
      <div class="mini">
        Body: {est["body_hours"]}h ({est["labor_body"]}) · Paint: {est["paint_hours"]}h ({est["labor_paint"]})<br/>
        Materials: {est["materials"]} · Parts (rough): {est["parts_cost"]}
      </div>
    </div>

    <div class="block">
      <div class="label">Photos</div>
      <div class="photos">{photo_links}</div>
    </div>

    <a class="cta" href="/quote/{shop_slug}">Back to start</a>

    <div class="note">Preliminary estimate. Final pricing after in-person inspection.</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
