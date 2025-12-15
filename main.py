import asyncio
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

SHOPS = {
    "miss": "Mississauga Collision Center",
    "mississauga-collision-center": "Mississauga Collision Center",
    "mississauga-collision-centre": "Mississauga Collision Centre",
}

def resolve_shop(key: Optional[str]) -> str:
    if not key:
        return "AI Estimate"
    k = key.lower().strip()
    return SHOPS.get(k, k.replace("-", " ").title())

@app.get("/api/health")
def health():
    return {"status": "ok"}

# ============================================================
# LANDING
# ============================================================
@app.get("/quote")
def landing(shop_id: Optional[str] = None):
    key = shop_id or "miss"
    name = resolve_shop(key)
    return HTMLResponse(render_landing(name, key))

@app.get("/quote/{slug}")
def landing_slug(slug: str):
    return HTMLResponse(render_landing(resolve_shop(slug), slug))

def render_landing(shop_name: str, shop_key: str) -> str:
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{}</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
<div class="page">
  <div class="card hero">
    <img src="/static/logo.png" class="logo landing-logo"/>
    <h1 class="title">{}</h1>
    <div class="subtitle">Upload photos to get a fast AI repair estimate.</div>
    <a href="/estimate?shop_key={}" class="cta">Start Estimate</a>
  </div>
</div>
</body>
</html>
""".format(shop_name, shop_name, shop_key)

# ============================================================
# UPLOAD
# ============================================================
@app.get("/estimate")
def upload(shop_key: Optional[str] = None):
    key = shop_key or "miss"
    return HTMLResponse(render_upload(resolve_shop(key), key))

def render_upload(shop_name: str, shop_key: str) -> str:
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{}</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
<div class="page">
  <div class="card hero">
    <h1 class="title">{}</h1>
    <form id="estimateForm">
      <input type="file" name="photo" required/>
      <button id="submitBtn" class="cta">Get Estimate</button>
    </form>
    <div id="loader" class="hidden">
      <div class="progress"><div id="progressBar" class="fill"></div></div>
    </div>
  </div>
</div>

<script>
document.getElementById("estimateForm").addEventListener("submit", async function(e) {{
  e.preventDefault();
  document.getElementById("loader").classList.remove("hidden");
  const bar = document.getElementById("progressBar");
  let p = 0;
  const t = setInterval(function() {{
    p += 10;
    bar.style.width = p + "%";
  }}, 500);

  const fd = new FormData(e.target);
  const r = await fetch("/api/estimate", {{ method: "POST", body: fd }});
  const j = await r.json();
  clearInterval(t);
  window.location = "/estimate/result?id=" + j.estimate_id;
}});
</script>
</body>
</html>
""".format(shop_name, shop_name)

# ============================================================
# RESULT
# ============================================================
ESTIMATES = {}

@app.get("/estimate/result")
def result(id: str):
    d = ESTIMATES.get(id)
    if not d:
        return HTMLResponse("Not found")

    items = "".join("<li>{}</li>".format(i) for i in d["areas"])

    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
<div class="page">
  <div class="card hero">
    <h1 class="title">AI Estimate</h1>
    <ul>{}</ul>
    <div class="big">${} – ${}</div>
  </div>
</div>
</body>
</html>
""".format(items, d["min"], d["max"]))

# ============================================================
# API
# ============================================================
@app.post("/api/estimate")
async def estimate(photo: UploadFile = File(...)):
    await asyncio.sleep(1)
    i = str(uuid.uuid4())
    ESTIMATES[i] = {
        "areas": ["Bumper", "Fender", "Headlight"],
        "min": 5200,
        "max": 7800
    }
    return {"estimate_id": i}
