import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# =============================
# CONFIG
# =============================

DEFAULT_SHOP_SLUG = "mississauga-collision-center"

SHOP_SLUG_MAP = {
    "mississauga-collision-center": {
        "id": "miss",
        "name": "Mississauga Collision Center"
    }
}

# =============================
# HEALTH CHECK
# =============================

@app.get("/api/health")
def health():
    return {"status": "ok"}

# =============================
# ROOT REDIRECT
# =============================

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(f"/quote/{DEFAULT_SHOP_SLUG}")

# =============================
# LEGACY QUERY SUPPORT
# =============================

@app.get("/quote", include_in_schema=False)
def legacy_quote(shop_id: str | None = None):
    return RedirectResponse(f"/quote/{DEFAULT_SHOP_SLUG}")

# =============================
# SLUG-BASED QUOTE ROUTE
# =============================

@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_by_slug(shop_slug: str):
    shop = SHOP_SLUG_MAP.get(shop_slug)
    if not shop:
        raise HTTPException(status_code=404, detail="Shop not found")

    shop_name = shop["name"]

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{shop_name} – AI Damage Estimator</title>

<style>
body {{
    background: radial-gradient(1200px 600px at top, #1b2b4f, #060b1a);
    color: white;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
}}

.container {{
    max-width: 420px;
    margin: 48px auto;
    padding: 28px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 22px;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.45);
}}

.logo {{
    width: 76px;
    height: 76px;
    border-radius: 50%;
    margin: 0 auto 14px;
    display: block;
}}

h1 {{
    text-align: center;
    font-size: 22px;
    margin-bottom: 14px;
}}

p {{
    text-align: center;
    opacity: 0.9;
}}

.button {{
    width: 100%;
    padding: 14px;
    font-size: 16px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #4f7cff, #7a6cff);
    color: white;
    cursor: pointer;
    margin-top: 20px;
}}

.note {{
    text-align: center;
    font-size: 13px;
    opacity: 0.7;
    margin-top: 12px;
}}
</style>
</head>

<body>
<div class="container">
    <img src="/logo.png" class="logo" />
    <h1>{shop_name}</h1>
    <p>Upload photos to get a fast AI repair estimate.</p>
    <button class="button">Start Estimate</button>
    <div class="note">
        Preliminary estimate · Final pricing after inspection
    </div>
</div>
</body>
</html>
"""

# =============================
# STATIC FILES (LOGO)
# =============================

if os.path.exists("logo.png"):
    app.mount("/", StaticFiles(directory=".", html=False), name="static")
