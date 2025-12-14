import os
import re
import secrets
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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

# Demo-only in-memory session store (token -> filenames)
SESSIONS: Dict[str, Dict] = {}

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "")).strip("_")
    return (name[:120] or "photo.jpg")

# ===============================
# Health check
# ===============================
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ===============================
# Landing page (per shop)
# ===============================
@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    shop_names = {
        "mississauga-collision-center": "Mississauga Collision Center"
    }

    shop_name = shop_names.get(
        shop_slug,
        shop_slug.replace("-", " ").title()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>{shop_name} – SimpleQuotez</title>
  <link rel="stylesheet" href="/static/style.css?v=UPLOAD2">
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
    shop_name = shop_slug.replace("-", " ").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>Upload Photos – {shop_name}</title>
  <link rel="stylesheet" href="/static/style.css?v=UPLOAD2">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>Upload Photos</h1>
    <p class="subtitle">Add 1–3 photos of the damage.</p>

    <form class="form" action="/quote/{shop_slug}/upload" method="post" enctype="multipart/form-data">
      <input class="file" type="file" name="photos" accept="image/*" multiple required />
      <button type="submit">Continue</button>
    </form>

    <div class="note">Tip: Overall shot + close-up = best results.</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)

# ===============================
# Upload handler
# ===============================
@app.post("/quote/{shop_slug}/upload")
async def upload_post(shop_slug: str, photos: List[UploadFile] = File(...)):
    if not photos:
        raise HTTPException(status_code=400, detail="Upload at least 1 photo.")

    # cap at 3 for now
    if len(photos) > 3:
        photos = photos[:3]

    token = secrets.token_urlsafe(18)
    folder = os.path.join(UPLOAD_DIR, token)
    os.makedirs(folder, exist_ok=True)

    filenames = []
    for i, up in enumerate(photos):
        fn = safe_filename(up.filename or f"photo_{i+1}.jpg")
        path = os.path.join(folder, fn)
        content = await up.read()
        with open(path, "wb") as f:
            f.write(content)
        filenames.append(fn)

    SESSIONS[token] = {
        "shop_slug": shop_slug,
        "filenames": filenames,
        "created_at": datetime.utcnow().isoformat(),
    }

    return RedirectResponse(url=f"/quote/{shop_slug}/uploaded/{token}", status_code=303)

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
# Uploaded confirmation screen
# ===============================
@app.get("/quote/{shop_slug}/uploaded/{token}", response_class=HTMLResponse)
def uploaded_page(shop_slug: str, token: str):
    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=404, detail="Session expired.")

    links = "".join(
        f'<a class="photo-link" href="/u/{token}/{fn}" target="_blank" rel="noopener">View photo {i+1}</a>'
        for i, fn in enumerate(sess["filenames"])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport"
        content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />
  <title>Uploaded</title>
  <link rel="stylesheet" href="/static/style.css?v=UPLOAD2">
</head>
<body>
  <div class="card">
    <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />
    <h1>Photos received ✅</h1>
    <p class="subtitle">Next we’ll run the estimate.</p>

    <div class="block">
      <div class="label">Uploaded photos</div>
      <div class="photos">{links}</div>
    </div>

    <a class="cta" href="/quote/{shop_slug}">Back</a>

    <div class="note">Next step (coming): AI estimate page.</div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
