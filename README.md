# AUTO-SHOP-V4 – Backend Only (Railway + Twilio)

This is a **backend-only** FastAPI service that:

- Receives SMS + photos from Twilio
- Uses OpenAI vision to estimate Ontario 2025 repair costs
- Saves estimates into Postgres
- Lets you view estimates through a simple admin API

## Files

- `main.py` – FastAPI app + Twilio webhook + AI estimator + DB models
- `requirements.txt` – Python dependencies
- `Procfile` – Start command for Railway (uvicorn)

## Environment variables (Railway → Service → Variables)

Create these variables on your **AUTO-SHOP** backend service:

- `OPENAI_API_KEY` – your OpenAI API key (project key)
- `DATABASE_URL` – Postgres URL from your Railway Postgres plugin
- `ADMIN_API_KEY` – make up a strong key, e.g. `sj_admin_XXXX`
- `SHOPS_JSON` – configuration for your shop(s), e.g.:

```json
[
  {
    "id": "sj_auto_body",
    "name": "SJ Auto Body",
    "webhook_token": "shop_sj_84k2p1"
  }
]
```

## Twilio Webhook URL

On your Twilio phone number, set **Messaging → A message comes in → Webhook** to:

```text
https://YOUR-RAILWAY-URL/sms-webhook?token=shop_sj_84k2p1
```

Replace `YOUR-RAILWAY-URL` with the `*.up.railway.app` domain from Railway.

## Starting locally (optional)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
