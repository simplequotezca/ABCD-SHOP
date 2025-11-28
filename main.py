import logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

# Use uvicorn logger so messages show up in Railway "Deploy Logs"
logger = logging.getLogger("uvicorn.error")

app = FastAPI()


@app.get("/", response_class=PlainTextResponse)
async def root():
    # Simple health check for browser
    return PlainTextResponse("OK")


@app.post("/sms-webhook", response_class=PlainTextResponse)
async def sms_webhook(request: Request):
    try:
        form = await request.form()
        body = (form.get("Body") or "").strip()
        from_number = form.get("From") or ""

        logger.info(f"📩 Incoming SMS from {from_number}: {body!r}")

        resp = MessagingResponse()
        resp.message(f"Auto-Repair bot received: '{body}' from {from_number}")

        # Twilio expects XML
        twiml = str(resp)
        logger.info(f"✅ Replying to Twilio with TwiML: {twiml!r}")

        return PlainTextResponse(twiml, media_type="application/xml")

    except Exception as e:
        # Even on error, reply with something so Twilio gets 200
        logger.exception("❌ Error in /sms-webhook handler: %s", e)
        resp = MessagingResponse()
        resp.message("Sorry, something went wrong on our end.")
        return PlainTextResponse(str(resp), media_type="application/xml")
