import os
import re
import json
import uuid
import time
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI
from PIL import Image

# ============================================================
# App
# ============================================================

app = FastAPI()

# Static (CSS + logo)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = Path("/tmp/simplequotez_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Shop config (expand later)
# ============================================================

SHOPS = [
    {
        "slug": "mississauga-collision-center",
        "name": "Mississauga Collision Center",
        "pricing": {
            "labor_rates": {"body": 95, "paint": 105},
            "materials_rate": 38,
            "base_floor": {
                "minor_min": 350,
                "minor_max": 650,
                "moderate_min": 900,
                "moderate_max": 1600,
                "severe_min": 2000,
                "severe_max": 5000,
            },
        },
    }
]

# ============================================================
# Session store (in-memory)
# token -> { status, created_at, shop_slug, filenames, ai, est, error }
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 30  # 30 minutes


def now_ts() -> float:
    return time.time()


def clean_old_sessions():
    cutoff = now_ts() - SESSION_TTL_SECONDS
    to_delete = []
    for token, s in SESSIONS.items():
        if s.get("created_at", 0) < cutoff:
            to_delete.append(token)
    for t in to_delete:
        # try delete folder too
        try:
            folder = UPLOAD_DIR / t
            if folder.exists():
                for p in folder.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    folder.rmdir()
                except Exception:
                    pass
        except Exception:
            pass
        SESSIONS.pop(t, None)


def get_shop(shop_slug: str) -> Dict[str, Any]:
    for s in SHOPS:
        if s["slug"] == shop_slug:
            return s
    # fallback: title-case slug
    return {
        "slug": shop_slug,
        "name": shop_slug.replace("-", " ").title(),
        "pricing": {
            "labor_rates": {"body": 95, "paint": 105},
            "materials_rate": 38,
            "base_floor": {
                "minor_min": 350,
                "minor_max": 650,
                "moderate_min": 900,
                "moderate_max": 1600,
                "severe_min": 2000,
                "severe_max": 5000,
            },
        },
    }


# ============================================================
# Utilities
# ============================================================

def html_escape(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def normalize_area_label(label: str) -> str:
    # remove weird slashes and repeated spaces
    label = (label or "").replace("\\", " ").replace("/", " ").strip()
    label = re.sub(r"\s+", " ", label)
    return label


def compress_image_to_jpeg_bytes(path: Path, max_side: int = 1280, quality: int = 82) -> bytes:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def b64_data_url_jpeg(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# Estimation logic (server-side guardrails)
# ============================================================

PART_BASE_COSTS = {
    "front bumper": 900,
    "rear bumper": 900,
    "bumper": 900,
    "fender": 700,
    "front fender": 700,
    "hood": 1100,
    "headlight": 1200,
    "headlight assembly": 1200,
    "grille": 600,
    "radiator support": 800,
    "door": 900,
    "quarter panel": 1200,
    "mirror": 350,
    "wheel": 450,
    "suspension": 1200,
    "cooling": 900,
    "sensor": 650,
    "adas": 0,  # calibration is labor/service not parts
}

HIDDEN_DAMAGE_MULTIPLIER = {
    "low": 1.05,
    "medium": 1.18,
    "high": 1.35,
}

SEVERITY_MULTIPLIER = {
    "minor": 1.0,
    "moderate": 1.35,
    "severe": 1.8,
}


def estimate_from_ai(shop: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    pricing = shop["pricing"]
    body_rate = float(pricing["labor_rates"]["body"])
    paint_rate = float(pricing["labor_rates"]["paint"])

    severity = (ai.get("severity") or "moderate").strip().lower()
    if severity not in ("minor", "moderate", "severe"):
        severity = "moderate"

    confidence = (ai.get("confidence") or "Medium").strip()
    hidden_risk = (ai.get("hidden_damage_risk") or "Medium").strip().lower()
    if hidden_risk not in ("low", "medium", "high"):
        hidden_risk = "medium"

    damaged = [normalize_area_label(x) for x in safe_list(ai.get("damaged_areas"))]
    ops = [normalize_area_label(x) for x in safe_list(ai.get("recommended_ops"))]

    # If AI did not provide hours, derive a sane baseline from severity & damaged panels
    n_panels = max(1, len(damaged))
    body_hours = ai.get("body_hours")
    paint_hours = ai.get("paint_hours")

    try:
        body_hours = float(body_hours)
    except Exception:
        # baseline: minor 6-10, moderate 10-18, severe 18-30
        if severity == "minor":
            body_hours = 6 + min(4, n_panels)
        elif severity == "severe":
            body_hours = 18 + min(12, 2 * n_panels)
        else:
            body_hours = 10 + min(8, int(1.5 * n_panels))

    try:
        paint_hours = float(paint_hours)
    except Exception:
        # baseline paint: 3-4 minor, 4-7 moderate, 6-10 severe
        if severity == "minor":
            paint_hours = 3.5
        elif severity == "severe":
            paint_hours = 6.5 + min(3.5, 0.5 * n_panels)
        else:
            paint_hours = 4.0 + min(3.0, 0.4 * n_panels)

    # Parts rough estimate
    parts_cost = ai.get("parts_cost")
    try:
        parts_cost = float(parts_cost)
    except Exception:
        rough = 0.0
        joined = " | ".join([d.lower() for d in damaged] + [o.lower() for o in ops])
        for key, cost in PART_BASE_COSTS.items():
            if key in joined:
                rough += float(cost)
        # keep a sane floor so estimates aren’t embarrassing
        if rough <= 0:
            rough = 1200.0 if severity != "minor" else 700.0
        parts_cost = rough

    # Materials (server-side)
    materials_cost = ai.get("materials_cost")
    try:
        materials_cost = float(materials_cost)
    except Exception:
        # simple materials baseline: scales with paint hours
        materials_cost = max(150.0, min(450.0, 110.0 + paint_hours * 22.0))

    # Multipliers
    sev_mult = SEVERITY_MULTIPLIER[severity]
    hidden_mult = HIDDEN_DAMAGE_MULTIPLIER[hidden_risk]

    body_labor = body_hours * body_rate
    paint_labor = paint_hours * paint_rate

    base_total = body_labor + paint_labor + materials_cost + parts_cost
    adjusted = base_total * sev_mult * hidden_mult

    # Range width grows with risk/severity (so customers aren’t shocked later)
    # Use ± 18% to 28%
    spread = 0.18
    if severity == "severe" or hidden_risk == "high":
        spread = 0.24
    if severity == "severe" and hidden_risk == "high":
        spread = 0.28

    min_total = max(350.0, adjusted * (1 - spread))
    max_total = adjusted * (1 + spread)

    # Round nicely
    def round_to(x: float, step: int = 10) -> int:
        return int(step * round(float(x) / step))

    min_total = round_to(min_total, 10)
    max_total = round_to(max_total, 10)

    return {
        "severity": severity.title(),
        "confidence": confidence,
        "hidden_damage_risk": hidden_risk.title(),
        "impact_zone": (ai.get("impact_zone") or "").strip(),
        "body_hours": round(float(body_hours), 1),
        "paint_hours": round(float(paint_hours), 1),
        "body_cost": round_to(body_labor, 10),
        "paint_cost": round_to(paint_labor, 10),
        "materials": round_to(materials_cost, 10),
        "parts": round_to(parts_cost, 10),
        "total_min": f"${min_total:,.0f}",
        "total_max": f"${max_total:,.0f}",
        "damaged_areas": damaged,
        "recommended_ops": ops,
    }


# ============================================================
# AI call (Vision)
# ============================================================

AI_SCHEMA_INSTRUCTIONS = """
Return ONLY valid JSON matching this schema (no extra text):
{
  "severity": "minor|moderate|severe",
  "confidence": "Low|Medium|High",
  "impact_zone": "e.g., front-left corner",
  "hidden_damage_risk": "Low|Medium|High",
  "damaged_areas": ["..."],
  "recommended_ops": ["..."],
  "notes": "2-4 sentences written for a customer",
  "body_hours": number,
  "paint_hours": number,
  "parts_cost": number
}
Rules:
- Use DRIVER'S POV for left/right (ex: driver's side = left in North America).
- If a front-corner impact is visible, set hidden_damage_risk to at least Medium.
- If headlight area is impacted, mention possible ADAS calibration in notes.
- If severity is severe OR hidden_damage_risk is High, include: "May exceed repair threshold after teardown." in notes.
- Keep damaged_areas and recommended_ops clean (no backslashes).
"""


async def run_ai_analysis(shop: Dict[str, Any], image_paths: List[Path]) -> Dict[str, Any]:
    # If no key, fail fast with a useful error
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Prepare images
    images_data_urls = []
    for p in image_paths[:3]:
        jpeg = compress_image_to_jpeg_bytes(p)
        images_data_urls.append(b64_data_url_jpeg(jpeg))

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert collision estimator. "
                "You will analyze vehicle damage from photos and return structured JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Shop context: {shop['name']} (Canada). {AI_SCHEMA_INSTRUCTIONS}"},
                *[{"type": "image_url", "image_url": {"url": url}} for url in images_data_urls],
            ],
        },
    ]

    # Use a vision-capable model available in OpenAI API
    # (Keep this stable; do not use experimental names)
    resp = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
    )
    text = (resp.choices[0].message.content or "").strip()

    # Extract JSON safely
    # Sometimes models wrap in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except Exception:
        # Try to find first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise RuntimeError("AI response was not valid JSON.")
        data = json.loads(m.group(0))

    # Normalize/clean for UI
    data["damaged_areas"] = [normalize_area_label(x) for x in safe_list(data.get("damaged_areas"))]
    data["recommended_ops"] = [normalize_area_label(x) for x in safe_list(data.get("recommended_ops"))]

    notes = (data.get("notes") or "").strip()
    if "teardown" not in notes.lower() and (
        str(data.get("severity", "")).lower() == "severe"
        or str(data.get("hidden_damage_risk", "")).lower() == "high"
    ):
        # enforce trust line
        notes = (notes + " " if notes else "") + "May exceed repair threshold after teardown."
    data["notes"] = notes

    # Enforce driver POV language in impact_zone if it contains raw "right/left" confusion:
    # (We don't rewrite everything, just keep it tidy)
    iz = (data.get("impact_zone") or "").strip()
    data["impact_zone"] = iz

    return data


# ============================================================
# Background task runner
# ============================================================

async def analyze_session(token: str):
    sess = SESSIONS.get(token)
    if not sess:
        return

    shop = get_shop(sess["shop_slug"])
    folder = UPLOAD_DIR / token
    image_paths = [folder / fn for fn in sess.get("filenames", [])]

    try:
        ai = await run_ai_analysis(shop, image_paths)
        est = estimate_from_ai(shop, ai)

        sess["ai"] = ai
        sess["est"] = est
        sess["status"] = "done"
        sess["error"] = ""
        sess["finished_at"] = now_ts()
    except Exception as e:
        sess["status"] = "error"
        sess["error"] = str(e)
        sess["finished_at"] = now_ts()


# ============================================================
# API
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status/{token}")
def status(token: str):
    clean_old_sessions()
    sess = SESSIONS.get(token)
    if not sess:
        return JSONResponse({"status": "missing"}, status_code=404)
    return {
        "status": sess.get("status", "processing"),
        "error": sess.get("error", "") if sess.get("status") == "error" else "",
    }


# ============================================================
# Pages
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    # default to first shop for demo
    return RedirectResponse(url=f"/quote/{SHOPS[0]['slug']}")


@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html_escape(shop_name)} - AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="shell">
    <div class="card hero">
      <img class="logo" src="/static/logo.png" alt="Simple Quotez" />
      <h1 class="title">{html_escape(shop_name)}</h1>
      <div class="subtitle">Upload photos to get a fast AI repair estimate.</div>

      <a class="cta" href="/quote/{html_escape(shop_slug)}/upload">Start Estimate</a>

      <div class="hint">
        <div class="hint-title">Best results with 3 photos:</div>
        <ul class="hint-list">
          <li>Overall damage</li>
          <li>Close-up</li>
          <li>Side angle</li>
        </ul>
      </div>

      <div class="fineprint">Preliminary range only. Final pricing is confirmed after teardown and in-person inspection.</div>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/quote/{shop_slug}/upload", response_class=HTMLResponse)
def upload_page(shop_slug: str):
    shop = get_shop(shop_slug)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Upload Photos - {html_escape(shop["name"])}</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="shell">
    <div class="card">
      <img class="logo" src="/static/logo.png" alt="Simple Quotez" />
      <h1 class="title">Upload Photos</h1>
      <div class="subtitle">Add 1–3 photos of the damage.</div>

      <form id="uploadForm" class="upload-form" action="/quote/{html_escape(shop_slug)}/upload" method="post" enctype="multipart/form-data">
        <div class="file-wrap">
          <input id="files" type="file" name="photos" accept="image/*" multiple required />
        </div>

        <button id="continueBtn" class="cta" type="submit">Continue</button>

        <div class="fineprint">Tip: Overall shot + close-up = best results.</div>
      </form>

      <a class="backlink" href="/quote/{html_escape(shop_slug)}">← Back</a>
    </div>
  </div>

<script>
(function() {{
  const form = document.getElementById('uploadForm');
  const btn = document.getElementById('continueBtn');
  let locked = false;

  form.addEventListener('submit', function(e) {{
    if (locked) {{
      e.preventDefault();
      return false;
    }}
    locked = true;
    btn.disabled = true;
    btn.classList.add('disabled');
    btn.textContent = 'Analyzing photos…';
  }});
}})();
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/quote/{shop_slug}/upload")
async def upload_photos(shop_slug: str, photos: List[UploadFile] = File(...)):
    clean_old_sessions()
    shop = get_shop(shop_slug)

    # Validate files
    if not photos or len(photos) < 1:
        raise HTTPException(status_code=400, detail="Please upload at least 1 photo.")
    if len(photos) > 3:
        photos = photos[:3]

    token = uuid.uuid4().hex
    folder = UPLOAD_DIR / token
    folder.mkdir(parents=True, exist_ok=True)

    filenames = []
    for idx, f in enumerate(photos):
        if not f.content_type or "image" not in f.content_type:
            continue
        ext = ".jpg"
        if f.filename and "." in f.filename:
            ext_guess = "." + f.filename.rsplit(".", 1)[-1].lower()
            if ext_guess in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ext_guess
        fn = f"photo_{idx+1}{ext}"
        out = folder / fn
        data = await f.read()
        out.write_bytes(data)
        filenames.append(fn)

    if not filenames:
        raise HTTPException(status_code=400, detail="Uploaded files were not valid images.")

    # Create session
    SESSIONS[token] = {
        "status": "processing",
        "created_at": now_ts(),
        "shop_slug": shop_slug,
        "filenames": filenames,
        "ai": None,
        "est": None,
        "error": "",
    }

    # Kick off background analysis (no waiting)
    asyncio.create_task(analyze_session(token))

    # Redirect to analyzing screen
    return RedirectResponse(url=f"/quote/{shop_slug}/analyzing/{token}", status_code=303)


@app.get("/quote/{shop_slug}/analyzing/{token}", response_class=HTMLResponse)
def analyzing_page(shop_slug: str, token: str):
    # If already finished, jump to results
    sess = SESSIONS.get(token)
    if sess and sess.get("status") == "done":
        return RedirectResponse(url=f"/quote/{shop_slug}/result/{token}")

    # On-brand analyzer page with polling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analyzing Photos</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="shell">
    <div class="card analyzer">
      <img class="logo" src="/static/logo.png" alt="Simple Quotez" />

      <h1 class="title">Analyzing Damage</h1>
      <div class="subtitle">This usually takes a few seconds.</div>

      <div class="progress">
        <div class="bar"><div class="fill" id="fill"></div></div>
        <div class="steps" id="stepText">Detecting damaged panels…</div>
      </div>

      <div class="fineprint">
        Please don’t refresh. We’ll automatically show your estimate as soon as it’s ready.
      </div>
    </div>
  </div>

<script>
(function() {{
  const steps = [
    "Detecting damaged panels…",
    "Estimating labor & parts…",
    "Assessing hidden damage risk…",
    "Finalizing estimate…"
  ];

  const fill = document.getElementById('fill');
  const stepText = document.getElementById('stepText');

  let stepIndex = 0;
  let pct = 6;

  // Smooth “real-feel” progress (caps at 92% until done)
  const tick = setInterval(() => {{
    pct += (pct < 70 ? 4 : (pct < 92 ? 1.5 : 0));
    if (pct > 92) pct = 92;
    fill.style.width = pct + "%";
  }}, 600);

  // Rotate step copy
  const stepTick = setInterval(() => {{
    stepIndex = (stepIndex + 1) % steps.length;
    stepText.textContent = steps[stepIndex];
  }}, 1500);

  // Poll status
  const started = Date.now();
  const minWaitMs = 2200; // prevents “blink” even if AI returns instantly

  async function poll() {{
    try {{
      const r = await fetch("/api/status/{token}", {{ cache: "no-store" }});
      if (!r.ok) throw new Error("status failed");
      const j = await r.json();

      if (j.status === "done") {{
        const elapsed = Date.now() - started;
        const waitMore = Math.max(0, minWaitMs - elapsed);
        setTimeout(() => {{
          fill.style.width = "100%";
          setTimeout(() => {{
            window.location.href = "/quote/{shop_slug}/result/{token}";
          }}, 250);
        }}, waitMore);
        clearInterval(tick);
        clearInterval(stepTick);
        return;
      }}

      if (j.status === "error") {{
        clearInterval(tick);
        clearInterval(stepTick);
        stepText.textContent = "We hit a snag. Please try again.";
        document.querySelector(".subtitle").textContent = (j.error || "Analysis failed.");
        fill.style.width = "100%";
        return;
      }}
    }} catch (e) {{
      // ignore transient network errors; keep polling
    }}

    setTimeout(poll, 900);
  }}

  poll();
}})();
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/quote/{shop_slug}/result/{token}", response_class=HTMLResponse)
def result_page(shop_slug: str, token: str):
    shop = get_shop(shop_slug)
    shop_name = shop["name"]

    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=404, detail="Session expired. Please upload again.")

    if sess.get("status") == "processing":
        return RedirectResponse(url=f"/quote/{shop_slug}/analyzing/{token}")

    if sess.get("status") == "error":
        err = html_escape(sess.get("error", "Analysis failed."))
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Estimate Error</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="shell">
    <div class="card">
      <img class="logo" src="/static/logo.png" alt="Simple Quotez" />
      <h1 class="title">Couldn’t Complete Analysis</h1>
      <div class="subtitle">{err}</div>
      <a class="cta" href="/quote/{html_escape(shop_slug)}/upload">Try Again</a>
      <a class="backlink" href="/quote/{html_escape(shop_slug)}">← Back</a>
    </div>
  </div>
</body>
</html>"""
        return HTMLResponse(html)

    ai = sess.get("ai") or {}
    est = sess.get("est") or {}

    damaged_areas = safe_list(est.get("damaged_areas") or ai.get("damaged_areas"))
    recommended_ops = safe_list(est.get("recommended_ops") or ai.get("recommended_ops"))

    damaged_html = "".join(f"<li>{html_escape(x)}</li>" for x in damaged_areas) or "<li>Inspection recommended</li>"
    ops_html = "".join(f"<li>{html_escape(x)}</li>" for x in recommended_ops) or "<li>In-person teardown & assessment</li>"

    severity = html_escape(est.get("severity", (ai.get("severity") or "Moderate").title()))
    confidence = html_escape(ai.get("confidence", est.get("confidence", "Medium")))
    risk = html_escape(est.get("hidden_damage_risk", ai.get("hidden_damage_risk", "Medium")))
    impact = html_escape(est.get("impact_zone", ai.get("impact_zone", "")))

    notes = (ai.get("notes") or "").strip()
    if not notes:
        notes = "Inspection recommended to confirm full scope and repairability."
    notes = html_escape(notes)

    total_min = html_escape(est.get("total_min", "$0"))
    total_max = html_escape(est.get("total_max", "$0"))

    body_hours = html_escape(est.get("body_hours", ""))
    paint_hours = html_escape(est.get("paint_hours", ""))
    body_cost = html_escape(est.get("body_cost", ""))
    paint_cost = html_escape(est.get("paint_cost", ""))
    materials = html_escape(est.get("materials", ""))
    parts = html_escape(est.get("parts", ""))

    # Photo links
    photo_links = ""
    for fn in sess.get("filenames", []):
        # We purposely don't expose raw file system; just a local route
        photo_links += f'<a class="photo-link" href="/uploads/{token}/{html_escape(fn)}" target="_blank">View photo</a>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Estimate - {html_escape(shop_name)}</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="shell">
    <div class="card">
      <img class="logo" src="/static/logo.png" alt="Simple Quotez" />
      <h1 class="title">AI Estimate</h1>

      <div class="pill">{severity} · {confidence} confidence</div>
      <div class="pill pill2">Hidden damage risk: {risk}{(" · " + impact) if impact else ""}</div>

      <div class="block">
        <div class="label">Summary</div>
        <div class="summary">{notes}</div>
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
        <div class="text big">{total_min} – {total_max}</div>

        <div class="subtle">
          Photo-based preliminary range. Final scope, parts, and pricing are confirmed after teardown and in-person inspection.
        </div>

        <div class="mini breakdown">
          Body labor: {body_hours}h ({("$" + str(body_cost)) if body_cost != "" else ""})<br/>
          Paint labor: {paint_hours}h ({("$" + str(paint_cost)) if paint_cost != "" else ""})<br/>
          Materials: {("$" + str(materials)) if materials != "" else ""}<br/>
          Parts (rough): {("$" + str(parts)) if parts != "" else ""}<br/>
          <div class="small-note">
            This estimate does not include unseen structural, suspension, cooling-system, or electronic damage that can be discovered after teardown.
          </div>
        </div>

        <div class="warning">
          <strong>Possible final repair cost may be higher.</strong><br/>
          Front-corner impacts often involve hidden damage (absorbers, brackets, radiator support, sensors, alignment, and calibration).
        </div>

        <div class="subtle bottom">
          This is an early guidance range to help you plan next steps. Final repairability and final price are confirmed after teardown and inspection.
        </div>
      </div>

      <div class="block">
        <div class="label">Photos</div>
        <div class="photos">{photo_links}</div>
      </div>

      <a class="cta" href="/quote/{html_escape(shop_slug)}">Start New Estimate</a>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


# ============================================================
# Serve uploaded files (read-only)
# ============================================================

@app.get("/uploads/{token}/{filename}", response_class=HTMLResponse)
def serve_upload(token: str, filename: str):
    folder = UPLOAD_DIR / token
    path = folder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    # Return raw bytes
    from fastapi.responses import Response
    content = path.read_bytes()
    # naive content-type
    ct = "image/jpeg"
    if filename.lower().endswith(".png"):
        ct = "image/png"
    elif filename.lower().endswith(".webp"):
        ct = "image/webp"
    return Response(content=content, media_type=ct)
