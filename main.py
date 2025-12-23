import os
import uuid
import json
import base64
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

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

    # Fallback: accept unknown shops without breaking the UI
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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_booking_email(
    shop_name: str,
    customer_name: str,
    phone: str,
    email: str,
    date: str,
    time: str,
    ai_summary: dict,
    to_email: str,
):
    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))

        subject = f"🛠 New Booking Request — {shop_name}"

        html = f"""
        <div style="font-family:Arial,sans-serif;line-height:1.5">
          <h2>New Booking Request</h2>

          <p><strong>Shop:</strong> {shop_name}</p>
          <hr/>

          <p><strong>Customer:</strong> {customer_name}</p>
          <p><strong>Phone:</strong> {phone}</p>
          <p><strong>Email:</strong> {email}</p>

          <hr/>

          <p><strong>Date:</strong> {date}</p>
          <p><strong>Time:</strong> {time}</p>

          <hr/>

          <p><strong>Severity:</strong> {ai_summary.get('severity')}</p>
          <p><strong>Confidence:</strong> {ai_summary.get('confidence')}</p>
          <p><strong>Labor:</strong> {ai_summary.get('labor_hours_range')}</p>
          <p><strong>Price:</strong> {ai_summary.get('price_range')}</p>

          <hr/>
          <p>Sent via <strong>SimpleQuotez AI Estimator</strong></p>
        </div>
        """

        message = Mail(
            from_email=os.getenv(
                "FROM_EMAIL",
                "AI Estimator – SimpleQuotez <simplequotez@yahoo.com>",
            ),
            to_emails=to_email,
            subject=subject,
            html_content=html,
        )

        sg.send(message)

    except Exception as e:
        print("SENDGRID ERROR:", repr(e))

# ============================================================
# AI VISION JSON CONTRACT
# ============================================================
AI_VISION_JSON_SCHEMA = {
    "name": "collision_estimate",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
            "severity": {"type": "string", "enum": ["Minor", "Moderate", "Severe"]},
            "driver_pov": {"type": "boolean"},
            "impact_side": {"type": "string", "enum": ["Driver", "Passenger", "Front", "Rear", "Unknown"]},
            "damaged_areas": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
            },
            "operations": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 18,
            },
            "structural_possible": {"type": "boolean"},
            "mechanical_possible": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "confidence",
            "severity",
            "driver_pov",
            "impact_side",
            "damaged_areas",
            "operations",
            "structural_possible",
            "mechanical_possible",
            "notes",
        ],
    },
}


# ============================================================
# RULE OVERRIDES (CLAMP AI OPTIMISM)
# ============================================================
def apply_rule_overrides(ai: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clamp optimism. Force safe operations/flags based on signals.
    """
    ai = dict(ai or {})

    # Normalize lists
    areas = ai.get("damaged_areas") or []
    ops = ai.get("operations") or []
    areas = [str(x).strip() for x in areas if str(x).strip()]
    ops = [str(x).strip() for x in ops if str(x).strip()]

    # If structural/mechanical possible => do not allow Minor
    if ai.get("structural_possible") or ai.get("mechanical_possible"):
        if ai.get("severity") == "Minor":
            ai["severity"] = "Moderate"

    # If frame/unibody keywords present => Severe
    frame_kw = ("frame", "unibody", "rail", "apron", "structure", "core support")
    if any(any(k in a.lower() for k in frame_kw) for a in areas):
        ai["severity"] = "Severe"
        ai["structural_possible"] = True

    # Mandatory ops for moderate+ in collision estimates
    must_ops = [
        "Pre-scan (diagnostics)",
        "Post-scan (diagnostics)",
        "Measure/inspect for hidden damage",
    ]
    for mo in must_ops:
        if mo not in ops:
            ops.append(mo)

    # Cap list sizes
    ai["damaged_areas"] = areas[:12]
    ai["operations"] = ops[:18]

    # Driver POV enforcement note (we require it)
    ai["driver_pov"] = True

    # Confidence normalization
    if ai.get("confidence") not in ("Low", "Medium", "High"):
        ai["confidence"] = "Medium"

    # Severity normalization
    if ai.get("severity") not in ("Minor", "Moderate", "Severe"):
        ai["severity"] = "Moderate"

    return ai


# ============================================================
# HOURS + COST LOGIC (RULES BASELINE + AI BLEND)
# ============================================================
OP_HOURS = {
    "Replace bumper": (3, 6),
    "Repair bumper": (2, 4),
    "Refinish bumper": (2, 3),
    "Replace fender": (3, 6),
    "Repair fender": (2, 5),
    "Replace headlight": (1, 2),
    "Repair hood": (2, 5),
    "Replace hood": (4, 7),
    "Inspect suspension": (1, 3),
    "Align / suspension check": (1, 2),
    "Pre-scan (diagnostics)": (0, 1),
    "Post-scan (diagnostics)": (0, 1),
    "Measure/inspect for hidden damage": (1, 2),
}


def rules_labor_range(operations: List[str], severity: str) -> Tuple[int, int]:
    mn = 0
    mx = 0
    for op in (operations or []):
        if op in OP_HOURS:
            a, b = OP_HOURS[op]
            mn += a
            mx += b
    # Safety minimums by severity
    if severity == "Minor":
        mn = max(mn, 3)
        mx = max(mx, 6)
    elif severity == "Moderate":
        mn = max(mn, 8)
        mx = max(mx, 14)
    else:
        mn = max(mn, 18)
        mx = max(mx, 33)
    return int(mn), int(mx)


def ai_adjusted_labor_range(severity: str, confidence: str) -> Tuple[float, float]:
    """
    Gentle AI-based adjustment target (not final).
    """
    base = {
        "Minor": (3.0, 6.0),
        "Moderate": (8.0, 14.0),
        "Severe": (18.0, 33.0),
    }.get(severity, (8.0, 14.0))

    # Confidence nudges
    conf_mult = {"Low": 0.9, "Medium": 1.0, "High": 1.05}.get(confidence, 1.0)
    return base[0] * conf_mult, base[1] * conf_mult


def blend_labor_ranges(rules_mn: int, rules_mx: int, ai_mn: float, ai_mx: float) -> Tuple[int, int]:
    """
    Blend AI influence at labor level only.
    """
    w = max(0.0, min(0.85, AI_LABOR_INFLUENCE))
    mn = (1 - w) * rules_mn + w * ai_mn
    mx = (1 - w) * rules_mx + w * ai_mx

    # Clamp to sane bounds
    mn = max(1.0, mn)
    mx = max(mn + 1.0, mx)

    # Keep modest spread
    if mx - mn > 22:
        mx = mn + 22

    return int(round(mn)), int(round(mx))


def risk_note_for(severity: str) -> str:
    if severity == "Severe":
        return "Hidden damage is common in hard impacts. Final cost may increase after teardown."
    if severity == "Minor":
        return "Final cost may vary after in-person inspection."
    return "Hidden damage is common in moderate impacts. Final cost may vary after inspection."


# ============================================================
# AI VISION CALL (HARDENED — NO MORE 500s)
# ============================================================
async def ai_vision_analyze(files: List[UploadFile]) -> Dict[str, Any]:
    fallback = {
        "confidence": "Medium",
        "severity": "Moderate",
        "driver_pov": True,
        "impact_side": "Unknown",
        "damaged_areas": ["Bumper", "Fender", "Headlight"],
        "operations": ["Replace bumper", "Repair fender", "Replace headlight"],
        "structural_possible": False,
        "mechanical_possible": False,
        "notes": "AI timeout or connection issue. Rules-based estimate used.",
    }

    if not client:
        # Fallback rules-only if no key
        return apply_rule_overrides({
            **fallback,
            "notes": "Fallback (no API key).",
        })

    image_parts = []
    for f in files:
        data = await f.read()
        if not data:
            continue

        # Hard cap to avoid transport errors on huge mobile photos
        if len(data) > 6_000_000:
            try:
                await f.seek(0)
            except Exception:
                pass
            continue

        try:
            await f.seek(0)
        except Exception:
            pass

        b64 = base64.b64encode(data).decode("utf-8")
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{f.content_type or 'image/jpeg'};base64,{b64}"},
            }
        )

    # If everything got skipped (too large / empty), do not call AI at all
    if not image_parts:
        return apply_rule_overrides({
            **fallback,
            "notes": "Images were too large or unreadable. Rules-based estimate used.",
        })

    prompt = (
        "You are an expert collision estimator.\n"
        "Analyze the vehicle damage from the photo(s).\n"
        "Output strictly valid JSON matching the provided schema.\n"
        "Use DRIVER'S POV for left/right.\n"
        "Be conservative: if unsure, choose a higher severity and flag structural/mechanical possible.\n"
        "Return operations as concise repair actions.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_parts,
                },
            ],
            response_format={"type": "json_schema", "json_schema": AI_VISION_JSON_SCHEMA},
            temperature=0.4,
        )

        raw = resp.choices[0].message.content or "{}"
        try:
            ai = json.loads(raw)
        except Exception:
            return apply_rule_overrides({
                **fallback,
                "notes": "AI response parsing failed. Rules-based estimate used.",
            })

        return apply_rule_overrides(ai)

    except Exception:
        # IMPORTANT: Never raise; never 500; always return a safe estimate
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
    request: Request,
    photos: List[UploadFile] = File(...),
    shop_key: str = Form("miss"),
):
    k, cfg = resolve_shop(shop_key)

    # limit 1-3 photos
    photos = (photos or [])[:3]

    ai = await ai_vision_analyze(photos)

    severity = ai.get("severity", "Moderate")
    confidence = ai.get("confidence", "Medium")

    damaged_areas = ai.get("damaged_areas", [])
    operations = ai.get("operations", [])

    rules_min, rules_max = rules_labor_range(operations, severity)
    ai_min_f, ai_max_f = ai_adjusted_labor_range(severity, confidence)
    hours_min, hours_max = blend_labor_ranges(rules_min, rules_max, ai_min_f, ai_max_f)

    labor_rate = int(cfg.get("labor_rate", SHOP_CONFIGS["miss"]["labor_rate"]))
    cost_min = hours_min * labor_rate
    cost_max = hours_max * labor_rate

    risk_note = risk_note_for(severity)

    summary = ai.get("notes", "").strip() or "Visible damage detected. Further inspection recommended."

    estimate_id = str(uuid.uuid4())

    # Store uploaded photos (in-memory) so shops can view them from the calendar event.
    stored_photos: List[bytes] = []
    for f in (photos or [])[:3]:
        try:
            # Ensure we read from the start (ai_vision_analyze already seeks, but keep robust)
            try:
                await f.seek(0)
            except Exception:
                pass
            b = await f.read()
            if b:
                stored_photos.append(b)
        except Exception:
            pass

    # Build absolute URLs for calendar/event description
    base = str(request.base_url).rstrip("/")
    photo_urls = [f"{base}/estimate/photo/{estimate_id}/{i}" for i in range(len(stored_photos))]

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
        "estimate_id": estimate_id,
        "photo_urls": photo_urls,
        "photos": stored_photos,  # bytes
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
        raise

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
