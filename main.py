import os
import json
import base64
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Optional OpenAI (graceful if missing)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

APP_TITLE = "SimpleQuotez Demo"
MAX_PHOTOS = 3
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# In-memory sessions for demo (Railway restarts clear this)
SESSIONS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title=APP_TITLE)

# Static files (expects ./static/style.css and ./static/logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Uploaded files (demo only)
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def html_escape(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))


def slugify(name: str) -> str:
    import re
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-") or "shop"


def _default_shops() -> List[Dict[str, Any]]:
    return [
        {
            "id": "miss",
            "name": "Mississauga Collision Centre",
            "webhook_token": "shop_miss_123",
            "calendar_id": "",
            "pricing": {
                "labor_rates": {"body": 95, "paint": 105},
                "materials_rate": 38,
                "base_floor": {
                    "minor_min": 350,
                    "minor_max": 650,
                    "moderate_min": 900,
                    "moderate_max": 1600,
                    "severe_min": 2000,
                    "severe_max": 5000
                }
            },
            "hours": {
                "monday": ["09:00", "18:00"],
                "tuesday": ["09:00", "18:00"],
                "wednesday": ["09:00", "18:00"],
                "thursday": ["09:00", "18:00"],
                "friday": ["09:00", "18:00"],
                "saturday": ["10:00", "15:00"],
                "sunday": None
            }
        }
    ]


def load_shops() -> List[Dict[str, Any]]:
    raw = os.getenv("SHOPS_JSON", "").strip()
    if not raw:
        shops = _default_shops()
    else:
        try:
            shops = json.loads(raw)
            if not isinstance(shops, list) or not shops:
                raise ValueError("SHOPS_JSON must be a non-empty list")
        except Exception as e:
            shops = _default_shops()
            shops[0]["_config_error"] = f"SHOPS_JSON parse error: {e}"

    for s in shops:
        if "slug" not in s or not s["slug"]:
            s["slug"] = slugify(s.get("name", "shop"))
    return shops


SHOPS = load_shops()


def get_shop(shop_slug: str) -> Dict[str, Any]:
    for s in SHOPS:
        if s.get("slug") == shop_slug or s.get("id") == shop_slug:
            return s
    fallback = _default_shops()[0]
    fallback["name"] = shop_slug.replace("-", " ").title()
    fallback["slug"] = shop_slug
    return fallback


@app.get("/api/health")
def health():
    return {"status": "ok"}


def estimate_from_ai(ai: Dict[str, Any], shop: Dict[str, Any]) -> Dict[str, Any]:
    pricing = shop.get("pricing", {}) or {}
    labor_rates = (pricing.get("labor_rates", {}) or {})
    rate_body = float(labor_rates.get("body", 95))
    rate_paint = float(labor_rates.get("paint", 105))
    materials_rate = float(pricing.get("materials_rate", 35))

    base = pricing.get("base_floor", {}) or {}
    sev = (ai.get("severity") or "Moderate").strip().lower()

    if "minor" in sev:
        floor_min, floor_max = float(base.get("minor_min", 350)), float(base.get("minor_max", 650))
    elif "severe" in sev or "major" in sev:
        floor_min, floor_max = float(base.get("severe_min", 2000)), float(base.get("severe_max", 5000))
    else:
        floor_min, floor_max = float(base.get("moderate_min", 900)), float(base.get("moderate_max", 1600))

    damaged = [str(x).lower() for x in (ai.get("damaged_areas") or [])]
    ops = [str(x).lower() for x in (ai.get("recommended_ops") or [])]

    body_hours = 0.0
    paint_hours = 0.0
    parts = 0.0

    def add_panel(panel: str, repair_h: float, replace_parts: float, paint_h: float):
        nonlocal body_hours, paint_hours, parts
        if any(panel in d for d in damaged):
            body_hours += repair_h
            paint_hours += paint_h
            if any(("replace" in o and panel in o) for o in ops):
                parts += replace_parts
                body_hours += max(1.5, repair_h * 0.6)

    add_panel("bumper", repair_h=3.5, replace_parts=650, paint_h=2.0)
    add_panel("fender", repair_h=4.0, replace_parts=480, paint_h=2.0)
    add_panel("hood", repair_h=5.0, replace_parts=850, paint_h=2.5)
    add_panel("headlight", repair_h=1.0, replace_parts=900, paint_h=0.0)
    add_panel("grille", repair_h=0.8, replace_parts=350, paint_h=0.0)
    add_panel("door", repair_h=4.5, replace_parts=900, paint_h=2.5)

    risk = (ai.get("hidden_damage_risk") or "").strip().lower()
    impact = (ai.get("impact_zone") or "").strip().lower()

    if "high" in risk:
        body_hours += 2.5
        parts += 400
    if "front" in impact:
        parts += 250

    notes = (ai.get("notes") or "").lower()
    if "adas" in notes or "calibrat" in notes:
        body_hours += 1.0
    if "alignment" in notes or "suspension" in notes:
        body_hours += 1.0

    materials = max(110.0, paint_hours * materials_rate)

    labor_body = body_hours * rate_body
    labor_paint = paint_hours * rate_paint
    subtotal = labor_body + labor_paint + materials + parts

    low = max(floor_min, subtotal * 0.90)
    high = max(low + 250.0, subtotal * 1.35)

    may_exceed = bool(ai.get("may_exceed_threshold")) or ("exceed" in notes)
    if not may_exceed:
        high = min(high, floor_max)
    else:
        high = max(high, floor_max * 1.25)

    def rnd(x: float) -> int:
        return int(round(x / 10.0) * 10)

    return {
        "total_min": rnd(low),
        "total_max": rnd(high),
        "body_hours": round(body_hours, 1),
        "paint_hours": round(paint_hours, 1),
        "labor_body": rnd(labor_body),
        "labor_paint": rnd(labor_paint),
        "materials": rnd(materials),
        "parts": rnd(parts),
        "rate_body": rate_body,
        "rate_paint": rate_paint,
    }


def openai_analyze(images_b64: List[str]) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.txt.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in Railway Variables.")
    client = OpenAI(api_key=api_key)

    instruction = """You are an expert collision estimator. From the photos, infer likely damaged areas and typical repair operations.
Return ONLY valid JSON with this schema:
{
  "severity": "Minor|Moderate|Severe",
  "confidence": "Low|Medium|High",
  "impact_zone": "front-left|front-right|rear-left|rear-right|side-left|side-right|unknown",
  "hidden_damage_risk": "Low|Medium|High",
  "damaged_areas": ["..."],
  "recommended_ops": ["..."],
  "notes": "1-3 short sentences. Include ADAS/alignment/teardown notes if relevant. Mention 'May exceed repair threshold after teardown' ONLY when appropriate.",
  "may_exceed_threshold": true|false
}
Rules:
- Always use driver perspective for left/right when possible.
- Use simple part names (bumper, fender, headlight, hood, door, grille, quarter panel, wheel, suspension, etc).
- If photo shows major structural / wheel / airbag / radiator support risk, choose Severe and hidden_damage_risk High.
"""

    content = [{"type": "input_text", "text": instruction}]
    for b64 in images_b64[:MAX_PHOTOS]:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": content}],
        temperature=0.2,
        max_output_tokens=600,
    )
    txt = resp.output_text.strip()

    try:
        data = json.loads(txt)
    except Exception:
        import re
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise RuntimeError("AI returned non-JSON output.")
        data = json.loads(m.group(0))

    data.setdefault("severity", "Moderate")
    data.setdefault("confidence", "Medium")
    data.setdefault("impact_zone", "unknown")
    data.setdefault("hidden_damage_risk", "Medium")
    data.setdefault("damaged_areas", [])
    data.setdefault("recommended_ops", [])
    data.setdefault("notes", "")
    data.setdefault("may_exceed_threshold", False)

    data["damaged_areas"] = list(dict.fromkeys([str(x).strip() for x in data["damaged_areas"] if str(x).strip()]))[:10]
    data["recommended_ops"] = list(dict.fromkeys([str(x).strip() for x in data["recommended_ops"] if str(x).strip()]))[:10]

    notes = str(data.get("notes", "")).strip()
    if not notes:
        notes = "Based on visible damage, an in-person inspection is recommended to confirm all required repairs."
    data["notes"] = notes

    return data


def build_shell(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html_escape(title)}</title>
  <link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
  <div class="page">
    <div class="card">
      {body_html}
    </div>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    shop = SHOPS[0]
    return RedirectResponse(url=f"/quote/{shop.get('slug','miss')}")


@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def landing(shop_slug: str):
    shop = get_shop(shop_slug)
    shop_name = shop.get("name", "Collision Center")
    cfg_err = shop.get("_config_error")

    err_html = f'<div class="warning"><strong>Config warning:</strong> {html_escape(cfg_err)}</div>' if cfg_err else ""

    body = f"""
      <img class="logo" src="/static/logo.png" alt="SimpleQuotez"/>
      <h1>{html_escape(shop_name)}</h1>
      <p class="subtitle">Upload photos to get a fast AI repair estimate.</p>

      {err_html}

      <a class="cta cta-glow" href="/quote/{html_escape(shop_slug)}/upload">Start Estimate</a>

      <div class="hint">
        <div class="hint-title">Best results with 3 photos:</div>
        <ul>
          <li>Overall damage</li>
          <li>Close-up</li>
          <li>Side angle</li>
        </ul>
      </div>

      <div class="note">
        Preliminary range only. Final pricing is confirmed after teardown and in-person inspection.
      </div>
    """
    return build_shell(f"{shop_name} • SimpleQuotez", body)


@app.get("/quote/{shop_slug}/upload", response_class=HTMLResponse)
def upload_page(shop_slug: str):
    shop = get_shop(shop_slug)
    shop_name = shop.get("name", "Collision Center")

    body = f"""
      <img class="logo" src="/static/logo.png" alt="SimpleQuotez"/>
      <h1>Upload Photos</h1>
      <p class="subtitle">Add 1–3 photos of the damage.</p>

      <form class="upload" action="/quote/{html_escape(shop_slug)}/upload" method="post" enctype="multipart/form-data">
        <input class="file" type="file" name="photos" accept="image/*" multiple required />
        <button class="cta cta-glow" type="submit">Continue</button>
      </form>

      <div class="note">
        Tip: Overall shot + close-up = best results.
      </div>
    """
    return build_shell(f"Upload • {shop_name}", body)


@app.post("/quote/{shop_slug}/upload")
async def handle_upload(shop_slug: str, photos: List[UploadFile] = File(...)):
    shop = get_shop(shop_slug)

    if not photos:
        raise HTTPException(status_code=400, detail="No photos uploaded.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(status_code=400, detail=f"Max {MAX_PHOTOS} photos.")

    token = uuid.uuid4().hex[:10]
    token_dir = os.path.join(UPLOAD_DIR, token)
    os.makedirs(token_dir, exist_ok=True)

    saved_files: List[str] = []
    images_b64: List[str] = []

    for i, f in enumerate(photos[:MAX_PHOTOS]):
        data = await f.read()
        if not data:
            continue

        ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        filename = f"photo_{i+1}{ext}"
        path = os.path.join(token_dir, filename)
        with open(path, "wb") as out:
            out.write(data)
        saved_files.append(filename)

        images_b64.append(base64.b64encode(data).decode("utf-8"))

    if not saved_files:
        raise HTTPException(status_code=400, detail="Uploaded files were empty.")

    try:
        ai = openai_analyze(images_b64)
        est = estimate_from_ai(ai, shop)
    except Exception as e:
        ai = {
            "severity": "Moderate",
            "confidence": "Low",
            "impact_zone": "unknown",
            "hidden_damage_risk": "Medium",
            "damaged_areas": ["front bumper"],
            "recommended_ops": ["inspect for hidden damage"],
            "notes": f"AI analysis unavailable: {e}",
            "may_exceed_threshold": True,
        }
        est = estimate_from_ai(ai, shop)

    trust_line = (
        "Photo-based preliminary range. Final scope, parts, and pricing are confirmed after teardown and in-person inspection."
    )
    price_framing = (
        "This estimate does not include unseen structural, suspension, cooling-system, or electronic damage that can be discovered after teardown."
    )

    SESSIONS[token] = {
        "created_at": datetime.utcnow().isoformat(),
        "shop_slug": shop_slug,
        "files": saved_files,
        "ai": ai,
        "est": est,
        "trust_language": trust_line,
        "price_framing": price_framing,
    }

    return RedirectResponse(url=f"/quote/{html_escape(shop_slug)}/result/{token}", status_code=303)


@app.get("/quote/{shop_slug}/result/{token}", response_class=HTMLResponse)
def result_page(shop_slug: str, token: str):
    shop = get_shop(shop_slug)
    shop_name = shop.get("name", "Collision Center")

    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found (it may have expired).")

    ai = sess.get("ai") or {}
    est = sess.get("est") or {}
    trust_line = sess.get("trust_language") or ""
    price_framing = sess.get("price_framing") or ""

    severity = html_escape(ai.get("severity", "Moderate"))
    confidence = html_escape(ai.get("confidence", "Medium"))
    risk = html_escape(ai.get("hidden_damage_risk", "Medium"))
    impact = html_escape(ai.get("impact_zone", "unknown"))

    damaged_html = "".join(f"<li>{html_escape(x)}</li>" for x in (ai.get("damaged_areas") or [])) or "<li>Inspection recommended</li>"
    ops_html = "".join(f"<li>{html_escape(x)}</li>" for x in (ai.get("recommended_ops") or [])) or "<li>Inspect and confirm scope</li>"

    photo_links = []
    for fn in sess.get("files", []):
        url = f"/uploads/{token}/{fn}"
        photo_links.append(f'<a class="photo-link" href="{url}" target="_blank" rel="noopener">View photo</a>')
    photos_html = " ".join(photo_links) if photo_links else '<span class="muted">No photos</span>'

    notes = html_escape(ai.get("notes", "")).strip()
    if not notes:
        notes = "Based on visible damage, an in-person inspection is recommended."

    total_min = int(est.get("total_min", 0))
    total_max = int(est.get("total_max", 0))

    body_hours = html_escape(est.get("body_hours", 0))
    paint_hours = html_escape(est.get("paint_hours", 0))
    labor_body = html_escape(est.get("labor_body", 0))
    labor_paint = html_escape(est.get("labor_paint", 0))
    materials = html_escape(est.get("materials", 0))
    parts = html_escape(est.get("parts", 0))

    exceed = bool(ai.get("may_exceed_threshold"))
    warn_html = ""
    if exceed:
        warn_html = """
        <div class="warning">
          <strong>Possible final repair cost may be higher.</strong><br/>
          Front-corner impacts often involve hidden damage (absorbers, brackets, radiator support, sensors, alignment, and calibration).
        </div>
        """

    body = f"""
      <img class="logo" src="/static/logo.png" alt="SimpleQuotez"/>
      <h1>AI Estimate</h1>

      <div class="pills">
        <div class="pill">{severity} • {confidence} confidence</div>
        <div class="pill pill2">Hidden damage risk: {risk} • {impact}</div>
      </div>

      <div class="block">
        <div class="label">Summary</div>
        <div class="text">{notes}</div>
      </div>

      <div class="grid2">
        <div class="block">
          <div class="label">Likely damaged areas</div>
          <ul class="list">{damaged_html}</ul>
        </div>

        <div class="block">
          <div class="label">Recommended operations</div>
          <ul class="list">{ops_html}</ul>
        </div>
      </div>

      <div class="block">
        <div class="label">Estimate (CAD)</div>
        <div class="price">${total_min:,} – ${total_max:,}</div>

        <div class="mini">{html_escape(trust_line)}</div>

        <div class="mini breakdown">
          Body labor: {body_hours}h (${labor_body})<br/>
          Paint labor: {paint_hours}h (${labor_paint})<br/>
          Materials: ${materials}<br/>
          Parts (rough): ${parts}<br/>
          {html_escape(price_framing)}
        </div>

        {warn_html}

        <div class="subtle">
          This is an early guidance range to help you plan next steps. Final repairability and final price are confirmed after teardown and inspection.
        </div>
      </div>

      <div class="block">
        <div class="label">Photos</div>
        <div class="photos">{photos_html}</div>
      </div>

      <a class="cta cta-glow" href="/quote/{html_escape(shop_slug)}/upload">Run another estimate</a>

      <div class="note">
        Next: booking + calendar sync once this demo is locked.
      </div>
    """
    return build_shell(f"AI Estimate • {shop_name}", body)


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nDisallow: /\n"
