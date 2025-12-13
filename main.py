from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files (CSS, logo)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/quote/{shop_slug}", response_class=HTMLResponse)
def quote_page(shop_slug: str):
    # Simple slug → display name mapping (expand later)
    shop_names = {
        "mississauga-collision-center": "Mississauga Collision Center"
    }

    shop_name = shop_names.get(
        shop_slug,
        shop_slug.replace("-", " ").title()
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <title>{shop_name} – SimpleQuotez</title>

    <!-- Cache-busted CSS -->
    <link rel="stylesheet" href="/static/style.css?v=4">
</head>
<body>

    <div class="card">
        <img src="/static/logo.png" class="logo" alt="SimpleQuotez Logo" />

        <h1>{shop_name}</h1>

        <p>Upload photos to get a fast AI repair estimate.</p>

        <button onclick="alert('Photo upload step coming next')">
            Start Estimate
        </button>

        <div class="note">
            Preliminary estimate · Final pricing after inspection
        </div>
    </div>

</body>
</html>
"""
    return HTMLResponse(content=html)


# Root redirect (optional safety)
@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(
        content="""
        <html>
            <head>
                <meta http-equiv="refresh" content="0; url=/quote/mississauga-collision-center">
            </head>
        </html>
        """
    )
