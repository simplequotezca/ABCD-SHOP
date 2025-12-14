from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ===============================
# Serve static files
# ===============================
app.mount("/static", StaticFiles(directory="static"), name="static")


# ===============================
# Health check
# ===============================
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ===============================
# Quote landing page (per shop)
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

    <!-- FORCE mobile sizing -->
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0, maximum-scale=1.0, viewport-fit=cover" />

    <title>{shop_name} – SimpleQuotez</title>

    <!-- ABSOLUTE STATIC PATH -->
    <link rel="stylesheet" href="/static/style.css?v=FINAL">
</head>

<body>
    <div class="card">
        <img src="/static/logo.png" alt="SimpleQuotez" class="logo" />

        <h1>{shop_name}</h1>

        <p class="subtitle">
            Upload photos to get a fast AI repair estimate.
        </p>

        <button onclick="startEstimate()">Start Estimate</button>

        <!-- PHOTO UPLOAD GUIDANCE (Option B) -->
        <div class="upload-hint">
            <strong>Best results with 3 photos:</strong>
            <ul>
                <li>Overall damage</li>
                <li>Close-up</li>
                <li>Side angle</li>
            </ul>
        </div>

        <div class="note">
            Preliminary estimate · Final pricing after inspection
        </div>
    </div>

    <script>
        function startEstimate() {{
            alert("Photo upload coming next");
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
