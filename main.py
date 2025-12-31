import os
import uuid
import json
import base64
import asyncio
import random
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from severity_engine import infer_visual_flags, calculate_severity

# OpenAI (requirements: openai==1.30.5)
from openai import OpenAI

# Calendar integration (Google Service Account)
from calendar_service import create_calendar_event

app = FastAPI()

# Serve static files (CSS + logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# ENV / CONFIG
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Shop config (add more shops here)
# NOTE: Keep Miss at 110 so your current $ ranges don't swing wildly.
SHOP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "miss": {"name": "Mississauga Collision Center", "labor_rate": 110},  # CAD/hr
}

SHOP_ALIASES: Dict[str, str] = {
    "miss": "miss",
    "mississauga-collision-center": "miss",
    "mississauga-collision-centre": "miss",
    "mississauga_collision_center": "miss",
}

# In-memory store for demo (replace with DB later)
ESTIMATES: Dict[str, Dict[str, Any]] = {}

# ============================================================
# SHOP RESOLUTION
# ============================================================
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


# ============================================================
# SENDGRID — BOOKING EMAIL (HELPER)
# ============================================================



def send_booking_email(
    shop_name: str,
    customer_name: str,
    phone: str,
    email: str,
    date: str,
    time: str,
    ai_summary: dict,
    request_url: str,
    to_email: str,
) -> None:
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Email
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        reply_to_email = os.getenv("DEMO_REPLY_EMAIL")

        subject = f"🛠 New Booking Request — {shop_name}"

       
        html = f"""
<div style="font-family: Arial, sans-serif; background:#0b0f14; color:#ffffff; padding:24px;">

  <h2 style="margin:0 0 12px 0;">🛠 New Booking Request</h2>

  <p style="margin:0 0 16px 0; color:#cfd6dd;">
    A customer submitted an AI estimate and requested an appointment.
  </p>

  <div style="background:#141a22; padding:16px; border-radius:10px; margin-bottom:18px;">
    <p><strong>Shop:</strong> {shop_name}</p>
    <p><strong>Customer:</strong> {customer_name}</p>
    <p><strong>Phone:</strong> {phone}</p>
    <p><strong>Email:</strong> {email}</p>
    <p><strong>Date:</strong> {date}</p>
    <p><strong>Time:</strong> {time}</p>
  </div>

  <div style="background:#141a22; padding:16px; border-radius:10px; margin-bottom:18px;">
    <p><strong>Severity:</strong> {ai_summary.get('severity')}</p>
    <p><strong>Confidence:</strong> {ai_summary.get('confidence')}</p>
    <p><strong>Estimated Labor:</strong> {ai_summary.get('labor_hours_range')}</p>
    <p><strong>Estimated Range:</strong> {ai_summary.get('price_range')}</p>
  </div>

  <div style="text-align:center; margin:24px 0;">
    <a href="{request_url}"
       style="
         display:inline-block;
         padding:14px 22px;
         background:#3fa9f5;
         color:#000;
         font-weight:700;
         border-radius:8px;
         text-decoration:none;
       ">
       View Full Estimate & Photos
    </a>
  </div>

  <p style="font-size:13px; color:#9aa4af; margin-top:24px;">
    This estimate is preliminary and based on uploaded photos.
    Final pricing is confirmed after teardown and in-person inspection.
  </p>

  <hr style="border:none; border-top:1px solid #222; margin:24px 0;" />

  <p style="font-size:12px; color:#6b7280;">
    Sent via <strong>SimpleQuotez AI Estimator</strong>
  </p>

</div>
"""

        message = Mail(
            from_email=Email("bookings@simplequotez.com", "SimpleQuotez"),
            to_emails=to_email,
            subject=subject,
            html_content=html,
        )

        if reply_to_email:
            message.reply_to = Email(reply_to_email)

        sg.send(message)

    except Exception as e:
        print("SENDGRID ERROR:", repr(e))


# ============================================================
# RULE OVERRIDES (CLAMP AI OPTIMISM)
# ============================================================
def apply_rule_overrides(ai: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize AI output & enforce conservative defaults.
    NOTE: This does NOT compute your final displayed severity — the severity_engine does.
    """
    ai = dict(ai or {})

    areas = ai.get("damaged_areas") or []
    ops = ai.get("operations") or []
    areas = [str(x).strip() for x in areas if str(x).strip()]
    ops = [str(x).strip() for x in ops if str(x).strip()]

    # Cap list sizes
    ai["damaged_areas"] = areas[:12]
    ai["operations"] = ops[:18]

    # Driver POV enforcement
    ai["driver_pov"] = True

    # Confidence normalization
    if ai.get("confidence") not in ("Low", "Medium", "High"):
        ai["confidence"] = "Medium"

    # impact_side normalization
    if not isinstance(ai.get("impact_side"), str) or not ai.get("impact_side"):
        ai["impact_side"] = "Unknown"

    # notes normalization
    if not isinstance(ai.get("notes"), str):
        ai["notes"] = ""

    # Boolean normalization
    ai["structural_possible"] = bool(ai.get("structural_possible", False))
    ai["mechanical_possible"] = bool(ai.get("mechanical_possible", False))

    # Mandatory ops (always useful on real estimates)
    must_ops = [
        "Pre-scan (diagnostics)",
        "Post-scan (diagnostics)",
        "Measure/inspect for hidden damage",
    ]
    for mo in must_ops:
        if mo not in ai["operations"]:
            ai["operations"].append(mo)

    return ai


# ============================================================
# RISK NOTE (MATCHES severity_engine LABELS)
# ============================================================
def risk_note_for(severity: str) -> str:
    s = (severity or "").lower()
    if "structural" in s:
        return "Hidden structural damage is possible. Final cost may increase after teardown."
    if "mechanical" in s:
        return "Mechanical / suspension involvement may be present. Final cost may vary after inspection."
    return "Final cost may vary after in-person inspection."


# ============================================================
# IMAGE PREPROCESSING (CRITICAL FOR MOBILE PHOTOS)
# ============================================================
MAX_AI_IMAGE_BYTES = int(os.getenv("MAX_AI_IMAGE_BYTES", "1500000"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))


def preprocess_image_for_ai(raw: bytes) -> bytes:
    if not raw:
        return b""

    try:
        img = Image.open(BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)

        if long_edge > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / long_edge
            img = img.resize((int(w * scale), int(h * scale)))

        quality = 85
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()

        while len(data) > MAX_AI_IMAGE_BYTES and quality > 35:
            quality -= 10
            out = BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            data = out.getvalue()

        return data

    except Exception as e:
        print("IMAGE_PREPROCESS_ERROR:", repr(e))
        return raw


# ============================================================
# AI VISION SCHEMA
# ============================================================
AI_VISION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "collision_estimate",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
            "impact_side": {"type": "string"},
            "driver_pov": {"type": "boolean"},
            "damaged_areas": {"type": "array", "items": {"type": "string"}},
            "operations": {"type": "array", "items": {"type": "string"}},
            "structural_possible": {"type": "boolean"},
            "mechanical_possible": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "confidence",
            "impact_side",
            "driver_pov",
            "damaged_areas",
            "operations",
            "structural_possible",
            "mechanical_possible",
            "notes",
        ],
    },
}

# ============================================================
# AI VISION CALL (HARDENED — NO MORE 500s)
# ============================================================
AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "12"))
AI_MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "2"))


async def ai_vision_analyze_bytes(images: List[bytes]) -> Dict[str, Any]:
    fallback = {
        "confidence": "Medium",
        "impact_side": "Unknown",
        "driver_pov": True,
        "damaged_areas": ["Bumper", "Fender", "Headlight"],
        "operations": ["Replace bumper", "Repair fender", "Replace headlight"],
        "structural_possible": False,
        "mechanical_possible": False,
        "notes": "Preliminary estimate based on visible damage patterns. Final cost may change after inspection.",
    }

    if not client:
        print("AI_DISABLED")
        return apply_rule_overrides(fallback)

    image_parts = []
    for b in images:
        if not b or len(b) > 6_000_000:
            continue
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"
                },
            }
        )

    if not image_parts:
        print("AI_SKIPPED_NO_IMAGES")
        return apply_rule_overrides(fallback)

    prompt = (
        "You are an expert collision estimator.\n"
        "Analyze vehicle damage from photos.\n"
        "Return ONLY valid JSON using the provided schema.\n"
        "Use driver's POV. Be conservative.\n"
        "If you can infer impact zone (front-left/front-right/rear-left/rear-right), put it in impact_side."
    )

    for attempt in range(AI_MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=VISION_MODEL,
                    messages=[
                        {"role": "system", "content": "Return ONLY JSON."},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}] + image_parts,
                        },
                    ],
                    response_format={"type": "json_schema", "json_schema": AI_VISION_JSON_SCHEMA},
                    temperature=0.3,
                ),
                timeout=AI_TIMEOUT_SECONDS,
            )

            ai = json.loads(resp.choices[0].message.content or "{}")
            print("AI_OK")
            return apply_rule_overrides(ai)

        except Exception as e:
            print("AI_ATTEMPT_FAILED:", repr(e))
            if attempt < AI_MAX_RETRIES:
                await asyncio.sleep(0.4 * (2**attempt) + random.uniform(0, 0.3))

    print("AI_FALLBACK")
    return apply_rule_overrides(fallback)


# ============================================================
# UI RENDERING (LOCKED STYLE)
# ============================================================
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
    <div class="subtitle">Upload 1–3 photos of the damage to receive a quick repair estimate.</div>

    <form id="estimateForm">
      <input type="hidden" name="shop_key" value="{shop_key}" />
      <input type="file" name="photos" accept="image/*" multiple required />
      <button class="cta" id="submitBtn" type="submit">Analyze photos</button>
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
      btn.textContent = 'Analyze photos';
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
    pill = f"Preliminary assessment: {data['severity']} • {data['confidence']} confidence"
    reasons_html = ""
    reasons = data.get("reasons") or []
    if reasons:
        reasons_html = f"""
        <div class="divider" style="margin-top:14px;"></div>
        <div style="margin-top:12px;">
          <div class="subtitle" style="margin-bottom:8px;">Inspection considerations</div>
          <ul style="margin-top:0;">{bullets(reasons)}</ul>
        </div>
        """

    impact_html = ""
    if data.get("impact_side"):
        impact_html = f"<div style='margin-top:10px;'>Impact zone: <strong>{data['impact_side']}</strong></div>"
        photo_html = ""
    photo_urls = data.get("photo_urls", [])

    if photo_urls:
        imgs = ""
        for url in photo_urls:
            imgs += f"""
              <a href="{url}" target="_blank">
                <img src="{url}"
                     style="
                       width:100%;
                       border-radius:10px;
                       margin-top:12px;
                       border:1px solid rgba(255,255,255,0.08);
                     " />
              </a>
            """

        photo_html = f"""
        <div class="divider" style="margin-top:18px;"></div>
        <div style="margin-top:14px;">
          <div class="subtitle" style="margin-bottom:6px;">
            Uploaded damage photos
          </div>
          {imgs}
        </div>
        """
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

    {impact_html}
    {photo_html}

    <div style="margin-top:14px; line-height:1.45;">
      {data["summary"]}
    </div>

    <ul style="margin-top:12px;">{bullets(data["damaged_areas"])}</ul>
    <ul style="margin-top:10px;">{bullets(data["operations"])}</ul>

    <div class="price-block">

  <div class="price-title">Initial intake assessment</div>

  <div class="price-label">Estimated labor cost (initial intake)</div>
  <div class="price-range">{data["cost_min"]} – {data["cost_max"]}</div>

  <div class="price-scope">
    Based on visible damage only.<br/>
    Parts, structural findings, and supplements are not included.
  </div>

  <div class="price-divider"></div>

  <div class="price-next-title">What happens next</div>
  <div class="price-next-text">
    This preliminary range helps the shop prepare for inspection and teardown.
    Final repair costs are determined after disassembly and diagnostics.
  </div>

  <div class="price-safety">
    Most collision repairs increase after teardown due to hidden damage.
  </div>

  </div>
    {reasons_html}

    <div class="divider" style="margin-top:16px;"></div>

    <div style="margin-top:14px;">
      <div class="subtitle" style="margin-bottom:10px;">Book an appointment</div>
      <form action="/book" method="post">
        <input type="hidden" name="estimate_id" value="{data["estimate_id"]}" />
        <input type="hidden" name="shop_key" value="{data["shop_key"]}" />

        <input type="text" name="name" placeholder="Full name" required />
        <input type="tel" name="phone" placeholder="Phone number" required />
        <input type="email" name="email" placeholder="Email" required />

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;">
          <input type="date" name="date" required />
          <input type="time" name="time" required />
        </div>

        <button class="cta" type="submit" style="margin-top:12px;">Confirm booking</button>
      </form>

      <div class="fineprint" style="margin-top:10px;">
        You'll receive confirmation after the shop reviews your request.
      </div>
    </div>

    <a class="backlink" href="/quote?shop_id={data["shop_key"]}">← Start over</a>
  </div>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(
        url="/quote/mississauga-collision-center",
        status_code=302
    )

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
    request: Request,
    photos: List[UploadFile] = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)

    # limit 1-3 photos
    photos = (photos or [])[:3]

    # READ + PREPROCESS PHOTOS ONCE
    raw_photos: List[bytes] = []
    for f in photos:
        try:
            b = await f.read()
            if b:
                raw_photos.append(b)
        except Exception:
            pass

    processed_photos: List[bytes] = []
    for b in raw_photos:
        pb = preprocess_image_for_ai(b)
        if pb:
            processed_photos.append(pb)

    # Run AI on processed images
    ai = await ai_vision_analyze_bytes(processed_photos)

    # ----------------------------
    # VISUAL CONTEXT ENRICHMENT
    # Only adds hints when supported by AI text content.
    # ----------------------------
    notes = (ai.get("notes") or "").strip()
    areas_text = " ".join(ai.get("damaged_areas", [])).lower()
    ops_text = " ".join(ai.get("operations", [])).lower()
    combined = f"{areas_text} {ops_text} {notes.lower()}".strip()

    # Impact side / zone (ONLY if the AI text actually contains it)
    if any(k2 in combined for k2 in ["left", "driver side", "lf", "front left", "front-left"]):
        notes = (notes + " Front-left impact.").strip()
    elif any(k2 in combined for k2 in ["right", "passenger side", "rf", "front right", "front-right"]):
        notes = (notes + " Front-right impact.").strip()

    # Offset front-corner heuristic (needs both bumper + (fender OR headlight))
    if ("bumper" in combined) and (("fender" in combined) or ("headlight" in combined)):
        notes = (notes + " Offset front-corner collision pattern.").strip()

    # Wheel involvement (ONLY if wheel/tire/rim appears)
    if any(k2 in combined for k2 in ["wheel", "tire", "rim"]):
        notes = (notes + " Wheel area involvement suspected.").strip()

    ai["notes"] = notes

    damaged_areas = ai.get("damaged_areas", [])
    operations = ai.get("operations", [])

    # === SEVERITY ENGINE (AUTHORITATIVE) ===
    flags = infer_visual_flags(ai)
    severity_data = calculate_severity(flags)

    severity = severity_data["severity"]
    confidence = severity_data["confidence"]
    hours_min, hours_max = severity_data["labor_range"]
    reasons = severity_data.get("reasons", [])

    labor_rate = int(cfg.get("labor_rate", SHOP_CONFIGS["miss"]["labor_rate"]))
    cost_min = hours_min * labor_rate
    cost_max = hours_max * labor_rate

    risk_note = risk_note_for(severity)
    summary = ai.get("notes", "").strip() or "Visible damage detected. Further inspection recommended."
    impact_side = ai.get("impact_side", "Unknown")

    estimate_id = str(uuid.uuid4())

    stored_photos = processed_photos[:3]
    base = os.getenv("PUBLIC_BASE_URL", str(request.base_url)).rstrip("/")
    photo_urls = [f"{base}/estimate/photo/{estimate_id}/{i}" for i in range(len(stored_photos))]
    request_url = f"{base}/estimate/result?id={estimate_id}"

    ESTIMATES[estimate_id] = {
        "shop_key": k,
        "severity": severity,
        "confidence": confidence,
        "impact_side": impact_side,
        "summary": summary,
        "damaged_areas": damaged_areas,
        "operations": operations,
        "reasons": reasons,
        "labour_hours_min": hours_min,
        "labour_hours_max": hours_max,
        "cost_min": money_fmt(cost_min),
        "cost_max": money_fmt(cost_max),
        "risk_note": risk_note,
        "estimate_id": estimate_id,
        "photo_urls": photo_urls,
        "photos": stored_photos,  # bytes
        "request_url": request_url,
    }

    return JSONResponse({"estimate_id": estimate_id})


@app.get("/estimate/photo/{estimate_id}/{idx}", include_in_schema=False)
def estimate_photo(estimate_id: str, idx: int):
    data = ESTIMATES.get(estimate_id)
    if not data:
        return Response(status_code=404)
    photos = data.get("photos") or []
    if idx < 0 or idx >= len(photos):
        return Response(status_code=404)
    return Response(content=photos[idx], media_type="image/jpeg")


@app.get("/estimate/result", response_class=HTMLResponse)
def estimate_result(id: str):
    data = ESTIMATES.get(id)
    if not data:
        return RedirectResponse(url="/quote?shop_id=miss")
    return HTMLResponse(render_result(data))


@app.post("/book", response_class=HTMLResponse)
def book_appointment(
    estimate_id: str = Form(...),
    shop_key: str = Form("miss"),
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
):
    est = ESTIMATES.get(estimate_id)
    if not est:
        return HTMLResponse("<h3>Estimate not found.</h3>", status_code=404)

    k, cfg = resolve_shop(shop_key)

    try:
        start_dt = datetime.fromisoformat(f"{date}T{time}")
    except Exception:
        return HTMLResponse("<h3>Invalid date/time.</h3>", status_code=400)

    end_dt = start_dt + timedelta(hours=1)

    ai_summary = {
        "severity": est.get("severity"),
        "confidence": est.get("confidence"),
        "labor_hours_range": f"{est.get('labour_hours_min')}–{est.get('labour_hours_max')} hrs",
        "price_range": f"{est.get('cost_min')} – {est.get('cost_max')}",
    }

    try:
        r = create_calendar_event(
            shop_key=k,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            summary=f"New Booking – {cfg.get('name','Shop')}",
            customer={"name": name, "phone": phone, "email": email},
            photo_urls=est.get("photo_urls", []),
            ai_summary=ai_summary,
        )
    except Exception as e:
        print("CALENDAR ERROR:", repr(e))
        r = {}

    send_booking_email(
       shop_name=cfg.get("name", "Collision Shop"),
       customer_name=name,
       phone=phone,
       email=email,
       date=date,
       time=time,
       ai_summary=ai_summary,
       request_url=est.get("request_url"),
       to_email=os.getenv("SHOP_NOTIFICATION_EMAIL", "shiran.bookings@gmail.com"),
    )
    link = r.get("htmlLink") if isinstance(r, dict) else ""

    return HTMLResponse(
        f"""
        <html>
        <head>
            <title>Booking Confirmed</title>
            <link rel="stylesheet" href="/static/style.css" />
        </head>
        <body class="page">
            <div class="card">
                <div class="title">Booking request sent</div>
                <div class="subtitle">{cfg.get('name')}</div>
                <p>The shop has received your booking request.</p>
                <a href="{link}" target="_blank">View in Google Calendar</a><br/>
                <a href="/quote?shop_id={k}">← Start over</a>
            </div>
        </body>
        </html>
        """
    )
