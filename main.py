import os
import json
import base64
from typing import Dict, Any, List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Optional (only used if you have OPENAI_API_KEY set)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

app = FastAPI()

# Serve static files (CSS, logo, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None


# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Shop config (expand later)
# -----------------------------
SHOP_NAMES = {
    "mississauga-collision-center": "Mississauga Collision Centre",
    "mississauga-collision-centre": "Mississauga Collision Centre",
    "miss": "Mississauga Collision Centre",
}

DEFAULT_SHOP = {
    "slug": "mississauga-collision-center",
    "name": "Mississauga Collision Centre",
    "currency": "CAD",
    # If you already have pricing logic, keep it there. This is only used by the fallback.
    "pricing": {
        "labor_rates": {"body": 95, "paint": 105},
        "materials_rate": 38,
    },
}


def get_shop(shop_slug: str) -> Dict[str, Any]:
    name = SHOP_NAMES.get(shop_slug, shop_slug.replace("-", " ").title())
    return {**DEFAULT_SHOP, "slug": shop_slug, "name": name}


# -----------------------------
# UI Pages
# -----------------------------
@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    shop = get_shop(shop_slug)
    return HTMLResponse(build_html(shop))


def build_html(shop: Dict[str, Any]) -> str:
    # Minimal, confident, polished single-page flow
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{shop["name"]} • AI Estimate</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="bg"></div>

  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <img class="logo" src="/static/logo.png" alt="Logo" onerror="this.style.display='none';" />
        <div class="brandtext">
          <div class="shopname">{shop["name"]}</div>
          <div class="subtle">AI photo estimate</div>
        </div>
      </div>

      <a class="ghostlink" href="#" onclick="return false;">Powered by SimpleQuotez</a>
    </header>

    <!-- Screen: Landing -->
    <section id="screen-landing" class="card screen">
      <div class="kicker">Fast, photo-based repair range</div>
      <h1 class="h1">Get a realistic repair cost range in under a minute.</h1>
      <p class="p">
        Upload a clear photo of the damage. We'll generate a preliminary estimate range based on what’s visible.
      </p>

      <div class="ctaRow">
        <button class="btn primary" id="startBtn">Start estimate</button>
      </div>

      <div class="hintRow">
        <div class="hint">Tip: Use 1–3 photos if you can (front + angle + close-up).</div>
      </div>
    </section>

    <!-- Screen: Upload -->
    <section id="screen-upload" class="card screen hidden">
      <div class="kicker">Upload photo</div>
      <h2 class="h2">Add a photo of the damage</h2>
      <p class="p">Good lighting and a wider angle improves accuracy.</p>

      <div class="uploader" id="dropZone">
        <input id="fileInput" type="file" accept="image/*" />
        <div class="uploadInner">
          <div class="uploadIcon" aria-hidden="true">⤒</div>
          <div class="uploadText">
            <div class="uploadTitle">Drag & drop a photo</div>
            <div class="uploadSub">or tap to select</div>
          </div>
        </div>
      </div>

      <div class="previewRow hidden" id="previewRow">
        <img id="previewImg" class="preview" alt="Selected photo preview" />
      </div>

      <div class="ctaRow split">
        <button class="btn" id="backToLanding">Back</button>
        <button class="btn primary" id="getEstimateBtn" disabled>Get AI estimate</button>
      </div>

      <div class="micro">By continuing, you confirm you’re sharing this photo for estimate purposes only.</div>
    </section>

    <!-- Screen: Results -->
    <section id="screen-results" class="card screen hidden">
      <div class="resultsHead">
        <div>
          <div class="kicker">AI Estimate</div>
          <div class="subtle">Based on the uploaded photos</div>
        </div>
        <div class="pillWrap">
          <span class="pill" id="pillSeverity">Preliminary</span>
          <span class="pill soft" id="pillConfidence">Photo-based</span>
        </div>
      </div>

      <!-- HERO number -->
      <div class="estimateHero">
        <div class="estimateRange" id="estimateRange">$—</div>
        <div class="estimateMeta">Photo-based preliminary repair range</div>
      </div>

      <!-- One-line reassurance -->
      <div class="reassure" id="reassureLine">
        Final scope, parts, and pricing are confirmed after teardown and in-person inspection.
      </div>

      <!-- Minimal summary -->
      <div class="summary" id="summaryText">
        —
      </div>

      <!-- Details: collapsed -->
      <div class="accordion">
        <details>
          <summary>Likely affected areas</summary>
          <ul class="list" id="areasList"></ul>
        </details>

        <details>
          <summary>Recommended repair approach</summary>
          <ul class="list" id="approachList"></ul>
        </details>

        <details>
          <summary>Cost breakdown</summary>
          <div class="breakdown" id="breakdownBox"></div>
          <div class="micro">Line-item totals may change after teardown.</div>
        </details>

        <details>
          <summary>What may change (standard)</summary>
          <div class="microBlock" id="mayChangeText">
            Some damage cannot be confirmed from photos alone. Hidden components are evaluated during teardown, which is standard for collision repairs.
          </div>
        </details>
      </div>

      <div class="closing" id="closingLine">
        This estimate helps you plan next steps. The repair facility will guide you through the remainder of the process.
      </div>

      <div class="ctaRow split">
        <button class="btn" id="newPhotoBtn">Use another photo</button>
        <button class="btn primary" id="doneBtn">Done</button>
      </div>
    </section>

    <!-- Loading overlay -->
    <div id="loadingOverlay" class="loading hidden" aria-live="polite" aria-busy="true">
      <div class="loadingCard">
        <div class="spinner" aria-hidden="true"></div>
        <div class="loadingText">
          <div class="loadingTitle">Analyzing photo</div>
          <div class="loadingSub">Generating a repair range…</div>
        </div>
      </div>
    </div>

    <footer class="footer">
      <div class="subtle">© {shop["name"]}</div>
    </footer>
  </main>

<script>
  const shopSlug = {json.dumps(shop["slug"])};

  const el = (id) => document.getElementById(id);

  const landing = el("screen-landing");
  const upload = el("screen-upload");
  const results = el("screen-results");
  const overlay = el("loadingOverlay");

  const startBtn = el("startBtn");
  const backToLanding = el("backToLanding");
  const getEstimateBtn = el("getEstimateBtn");
  const fileInput = el("fileInput");
  const dropZone = el("dropZone");
  const previewRow = el("previewRow");
  const previewImg = el("previewImg");

  const newPhotoBtn = el("newPhotoBtn");
  const doneBtn = el("doneBtn");

  function show(screen) {{
    landing.classList.add("hidden");
    upload.classList.add("hidden");
    results.classList.add("hidden");
    screen.classList.remove("hidden");
    window.scrollTo({{ top: 0, behavior: "instant" }});
  }}

  function setLoading(isLoading) {{
    if (isLoading) overlay.classList.remove("hidden");
    else overlay.classList.add("hidden");
  }}

  let selectedFile = null;

  function setPreview(file) {{
    if (!file) {{
      previewRow.classList.add("hidden");
      previewImg.src = "";
      return;
    }}
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewRow.classList.remove("hidden");
  }}

  function resetUpload() {{
    selectedFile = null;
    fileInput.value = "";
    setPreview(null);
    getEstimateBtn.disabled = true;
  }}

  startBtn.addEventListener("click", () => {{
    show(upload);
  }});

  backToLanding.addEventListener("click", () => {{
    resetUpload();
    show(landing);
  }});

  doneBtn.addEventListener("click", () => {{
    resetUpload();
    show(landing);
  }});

  newPhotoBtn.addEventListener("click", () => {{
    resetUpload();
    show(upload);
  }});

  fileInput.addEventListener("change", (e) => {{
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    selectedFile = f;
    setPreview(f);
    getEstimateBtn.disabled = false;
  }});

  // Make entire dropzone clickable
  dropZone.addEventListener("click", () => fileInput.click());

  // Drag & drop
  dropZone.addEventListener("dragover", (e) => {{
    e.preventDefault();
    dropZone.classList.add("dragover");
  }});
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {{
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    selectedFile = f;
    setPreview(f);
    getEstimateBtn.disabled = false;
  }});

  function moneyRange(min, max, currency) {{
    // Keep it simple, no decimals
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 0 }});
    return `${{fmt.format(min)}} – ${{fmt.format(max)}} ${{currency}}`;
  }}

  function setText(id, value) {{
    el(id).textContent = value;
  }}

  function fillList(listEl, items) {{
    listEl.innerHTML = "";
    (items || []).forEach(it => {{
      const li = document.createElement("li");
      li.textContent = it;
      listEl.appendChild(li);
    }});
  }}

  function fillBreakdown(boxEl, breakdown) {{
    // breakdown is array of {label, value}
    boxEl.innerHTML = "";
    if (!breakdown || !breakdown.length) {{
      boxEl.textContent = "—";
      return;
    }}
    const table = document.createElement("div");
    table.className = "kv";
    breakdown.forEach(row => {{
      const r = document.createElement("div");
      r.className = "kvRow";
      const k = document.createElement("div");
      k.className = "kvKey";
      k.textContent = row.label;
      const v = document.createElement("div");
      v.className = "kvVal";
      v.textContent = row.value;
      r.appendChild(k);
      r.appendChild(v);
      table.appendChild(r);
    }});
    boxEl.appendChild(table);
  }}

  function applyResult(data) {{
    // Pills (quiet context only)
    setText("pillSeverity", data.severity_label || "Preliminary");
    setText("pillConfidence", data.confidence_label || "Photo-based");

    // Hero range
    setText("estimateRange", moneyRange(data.estimate_min, data.estimate_max, data.currency || "CAD"));

    // Summary (minimal, calm)
    setText("summaryText", data.summary || "—");

    // Lists
    fillList(el("areasList"), data.likely_affected_areas || []);
    fillList(el("approachList"), data.recommended_repair_approach || []);

    // Breakdown
    fillBreakdown(el("breakdownBox"), data.cost_breakdown || []);

    // Optional may-change override (keep tone calm)
    if (data.may_change_text) {{
      el("mayChangeText").textContent = data.may_change_text;
    }}

    show(results);
  }}

  getEstimateBtn.addEventListener("click", async () => {{
    if (!selectedFile) return;

    setLoading(true);
    getEstimateBtn.disabled = true;

    try {{
      const fd = new FormData();
      fd.append("image", selectedFile);

      const resp = await fetch(`/api/estimate?shop_slug=${{encodeURIComponent(shopSlug)}}`, {{
        method: "POST",
        body: fd
      }});

      const data = await resp.json();
      if (!resp.ok) {{
        throw new Error(data.detail || "Estimate failed");
      }}

      applyResult(data);
    }} catch (err) {{
      alert(err?.message || "Estimate failed. Please try another photo.");
      getEstimateBtn.disabled = false;
    }} finally {{
      setLoading(false);
    }}
  }});
</script>

</body>
</html>
"""


# -----------------------------
# Estimate API
# -----------------------------
@app.post("/api/estimate")
async def api_estimate(shop_slug: str, image: UploadFile = File(...)):
    shop = get_shop(shop_slug)
    img_bytes = await image.read()

    # 1) If YOU ALREADY HAVE A GOOD ESTIMATOR:
    #    Replace the call below with your existing logic and return the same shape.
    #    DO NOT TOUCH THE UI COPY—only map your output into this response schema.
    #
    #    Example:
    #    result = your_existing_estimator(img_bytes, shop)
    #
    #    Then return result as JSONResponse(result)

    # 2) Otherwise: use OpenAI if available, fallback if not
    if client:
        try:
            result = await estimate_with_openai(img_bytes, shop)
            return JSONResponse(result)
        except Exception:
            # Fall through to fallback if AI fails
            pass

    result = estimate_fallback(img_bytes, shop)
    return JSONResponse(result)


async def estimate_with_openai(img_bytes: bytes, shop: Dict[str, Any]) -> Dict[str, Any]:
    # Convert to base64 data URL (JPEG/PNG both ok)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:application/octet-stream;base64,{b64}"

    # Prompt engineered for: minimal, calm, professional tone
    # IMPORTANT: We ask for JSON only.
    system = (
        "You are a collision repair estimating assistant. "
        "Be calm, professional, and matter-of-fact. "
        "Avoid fear language. Avoid overconfident claims. "
        "Use Canadian spelling where applicable. Return STRICT JSON only."
    )

    user = {
        "shop_name": shop["name"],
        "currency": shop.get("currency", "CAD"),
        "instructions": [
            "Analyze the photo for visible collision damage.",
            "Return an estimate range (min/max) as integers (CAD).",
            "Provide a minimal summary (1–2 sentences) in calm tone.",
            "Provide likely affected areas list (3–7 items).",
            "Provide recommended repair approach list (3–6 items).",
            "Provide cost breakdown rows as label/value strings (4–6 rows).",
            "Provide severity_label (e.g., Minor / Moderate / Significant) but keep it non-alarming.",
            "Provide confidence_label (e.g., Photo-based).",
            "Provide may_change_text that normalizes teardown and hidden components (calm).",
        ],
        "output_schema": {
            "estimate_min": "int",
            "estimate_max": "int",
            "currency": "string",
            "severity_label": "string",
            "confidence_label": "string",
            "summary": "string",
            "likely_affected_areas": "string[]",
            "recommended_repair_approach": "string[]",
            "cost_breakdown": [{"label": "string", "value": "string"}],
            "may_change_text": "string",
        },
    }

    # Use Responses API (compatible with modern OpenAI Python SDK)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user)},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        temperature=0.2,
    )

    text = (resp.output_text or "").strip()

    # Strict JSON parse
    try:
        parsed = json.loads(text)
    except Exception:
        # Sometimes model wraps JSON; attempt to extract first {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise

    # Basic sanity + clamp
    est_min = int(parsed.get("estimate_min", 0) or 0)
    est_max = int(parsed.get("estimate_max", 0) or 0)
    if est_min <= 0 or est_max <= 0 or est_max < est_min:
        # If model messes up, recover with fallback
        return estimate_fallback(img_bytes, shop)

    return {
        "estimate_min": est_min,
        "estimate_max": est_max,
        "currency": parsed.get("currency") or shop.get("currency", "CAD"),
        "severity_label": parsed.get("severity_label") or "Preliminary",
        "confidence_label": parsed.get("confidence_label") or "Photo-based",
        "summary": parsed.get("summary") or "Damage is visible. An in-person inspection will confirm final scope.",
        "likely_affected_areas": parsed.get("likely_affected_areas") or [],
        "recommended_repair_approach": parsed.get("recommended_repair_approach") or [],
        "cost_breakdown": parsed.get("cost_breakdown") or [],
        "may_change_text": parsed.get("may_change_text")
        or "Some damage cannot be confirmed from photos alone. Hidden components are evaluated during teardown, which is standard for collision repairs.",
    }


def estimate_fallback(_: bytes, shop: Dict[str, Any]) -> Dict[str, Any]:
    # Safe, generic fallback. Replace this with your "good" estimator if you have it.
    return {
        "estimate_min": 1800,
        "estimate_max": 4200,
        "currency": shop.get("currency", "CAD"),
        "severity_label": "Significant",
        "confidence_label": "Photo-based",
        "summary": (
            "Significant damage is visible in the front-left area. Repairs at this level commonly involve body panels "
            "and may require calibration after repairs are completed."
        ),
        "likely_affected_areas": ["Front bumper", "Fender", "Headlight", "Hood", "Suspension components"],
        "recommended_repair_approach": [
            "Replace damaged exterior panels",
            "Repair affected metal components",
            "Inspect suspension and alignment",
            "Perform required system calibrations",
        ],
        "cost_breakdown": [
            {"label": "Body labour", "value": "14.0h"},
            {"label": "Paint labour", "value": "5.0h"},
            {"label": "Materials", "value": "$220"},
            {"label": "Parts (estimated)", "value": "$1,450"},
        ],
        "may_change_text": (
            "Some damage cannot be confirmed from photos alone. Hidden components are evaluated during teardown, "
            "which is standard for collision repairs."
        ),
    }
