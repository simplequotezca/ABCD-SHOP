# deploy reset
import os
import re
import json
import uuid
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from pydantic import BaseModel
from openai import OpenAI

try:
    from twilio.twiml.messaging_response import MessagingResponse
except ImportError:
    class MessagingResponse:
        def __init__(self): self.msg=[]
        def message(self,t): self.msg.append(t)
        def __str__(self): return "<Response>" + "".join(f"<Message>{m}</Message>" for m in self.msg) + "</Response>"

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

class LaborRates(BaseModel):
    body: float
    paint: float

class BaseFloor(BaseModel):
    minor_min: int
    minor_max: int
    moderate_min: int
    moderate_max: int
    severe_min: int
    severe_max: int

class ShopPricing(BaseModel):
    labor_rates: LaborRates
    materials_rate: float
    base_floor: BaseFloor

class ShopHours(BaseModel):
    monday: List[str]
    tuesday: List[str]
    wednesday: List[str]
    thursday: List[str]
    friday: List[str]
    saturday: List[str]
    sunday: List[str]

class Shop(BaseModel):
    id: str
    name: str
    webhook_token: str
    calendar_id: str
    pricing: ShopPricing
    hours: ShopHours

def load_shops() -> Dict[str, Shop]:
    raw = os.getenv("SHOPS_JSON")
    if not raw: raise RuntimeError("SHOPS_JSON missing")
    data = json.loads(raw)
    out={}
    for s in data:
        shop = Shop(
            id=s["id"],
            name=s["name"],
            webhook_token=s["webhook_token"],
            calendar_id=s["calendar_id"],
            pricing=ShopPricing(**s["pricing"]),
            hours=ShopHours(**s["hours"])
        )
        out[shop.webhook_token] = shop
    return out

SHOPS_BY_TOKEN = load_shops()

SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 120

def sess_key(shop: Shop, phone: str): return f"{shop.id}:{phone}"

def get_session(shop: Shop, phone: str):
    key = sess_key(shop, phone)
    now = datetime.utcnow()
    s = SESSIONS.get(key)
    if s:
        try: created = datetime.fromisoformat(s["_created"])
        except: created = now
        if now - created > timedelta(minutes=SESSION_TTL):
            s=None
    if not s:
        s={"_created": now.isoformat()}
        SESSIONS[key]=s
    return s

def safe_json(raw):
    try: return json.loads(raw)
    except: return {}

def download_media(url: str) -> bytes:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    else:
        r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content

def to_data_url(data: bytes, ctype="image/jpeg"):
    return "data:%s;base64,%s" % (ctype, base64.b64encode(data).decode())
    # ============================================================
# Flexible date + time parsing (Option 1 — Full Natural Language)
# ============================================================

MONTHS = {
    "jan":1,"january":1, "feb":2,"february":2, "mar":3,"march":3,
    "apr":4,"april":4, "may":5,
    "jun":6,"june":6, "jul":7,"july":7,
    "aug":8,"august":8,
    "sep":9,"sept":9,"september":9,
    "oct":10,"october":10,
    "nov":11,"november":11,
    "dec":12,"december":12
}

def parse_time_any(text: str) -> Optional[tuple]:
    t = text.lower()

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        hour=int(m.group(1))
        minute=int(m.group(2) or "0")
        ap=m.group(3)
        if ap=="pm" and hour!=12: hour+=12
        if ap=="am" and hour==12: hour=0
        return hour,minute

    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h=int(m.group(1)); m2=int(m.group(2))
        if 0<=h<=23 and 0<=m2<=59:
            return h,m2

    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
    if m:
        hour=int(m.group(1)); minute=0
        ap=m.group(2)
        if ap=="pm" and hour!=12: hour+=12
        if ap=="am" and hour==12: hour=0
        return hour,minute

    return None

def parse_date_any(text: str) -> Optional[tuple]:
    t=text.lower().replace(","," ")
    t=re.sub(r"(\d+)(st|nd|rd|th)", r"\1", t)

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        month=int(m.group(1)); day=int(m.group(2))
        year=int(m.group(3)) if m.group(3) else datetime.utcnow().year
        if year<100: year+=2000
        return year,month,day

    m = re.search(r"\b([a-z]{3,9})\s+(\d{1,2})\b", t)
    if m:
        w=m.group(1); d=int(m.group(2))
        month=MONTHS.get(w[:3], MONTHS.get(w))
        if month: return datetime.utcnow().year,month,d

    m = re.search(r"\b(\d{1,2})\s+([a-z]{3,9})\b", t)
    if m:
        d=int(m.group(1)); w=m.group(2)
        month=MONTHS.get(w[:3], MONTHS.get(w))
        if month: return datetime.utcnow().year,month,d

    return None

def parse_datetime_any(text: str):
    missing=[]
    date_info=parse_date_any(text)
    time_info=parse_time_any(text)
    if not date_info: missing.append("date")
    if not time_info: missing.append("time")
    if date_info and time_info:
        y,m,d=date_info
        h,mi=time_info
        try: return datetime(y,m,d,h,mi), []
        except: return None, ["date"]
    return None, missing

def shop_open(shop: Shop, dt: datetime) -> bool:
    day = dt.strftime("%A").lower()
    hours: List[str] = getattr(shop.hours, day)
    if not hours or hours == ["closed"]: return False

    for block in hours:
        try:
            s,e = block.split("-")
            s=s.strip().lower(); e=e.strip().lower()

            def parse_block(t):
                m=re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
                if not m: return None
                h=int(m.group(1)); mi=int(m.group(2) or "0"); ap=m.group(3)
                if ap=="pm" and h!=12: h+=12
                if ap=="am" and h==12: h=0
                return dt.replace(hour=h,minute=mi,second=0,microsecond=0)

            sd=parse_block(s); ed=parse_block(e)
            if sd and ed and sd<=dt<=ed: return True
        except:
            continue
    return False


# ============================================================
# VIN DECODER (AI)
# ============================================================

VIN_PROMPT = """
You decode 17-character VINs.
Respond ONLY in JSON:
{
  "year": "...",
  "make": "...",
  "model": "...",
  "body_style": "..."
}
Use 'unknown' if not sure.
""".strip()

def decode_vin(vin: str) -> Dict[str,str]:
    c=client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":VIN_PROMPT},
            {"role":"user","content":vin}
        ]
    )
    out=safe_json(c.choices[0].message.content or "{}")
    out["vin"]=vin
    return out


# ============================================================
# PRE-SCAN (MULTI-IMAGE FUSION)
# ============================================================

PRESCAN_PROMPT = """
Fuse ALL photos. Identify damaged areas, damage types, and short notes.
Respond ONLY:
{
 "areas":[...],
 "damage_types":[...],
 "notes":[...]
}
""".strip()

def run_prescan(images: List[str]):
    content=[{"type":"text","text":"Fuse all photos and respond in JSON only."}]
    for url in images[:3]:
        content.append({"type":"image_url","image_url":{"url":url}})
    c=client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":PRESCAN_PROMPT},
            {"role":"user","content":content}
        ]
    )
    data=safe_json(c.choices[0].message.content or "{}")
    areas=[a.lower() for a in data.get("areas",[])]
    dmg=[d.lower() for d in data.get("damage_types",[])]
    notes=data.get("notes",[])
    if isinstance(notes,list): notes=" ".join(notes)
    return {"areas":areas,"damage_types":dmg,"notes":notes}


# ============================================================
# ESTIMATOR + LEVEL-C PRICING
# ============================================================

ESTIMATOR_PROMPT = """
You output severity, summary, and line_items. NO dollar amounts.
JSON only:
{
 "severity":"minor|moderate|severe|unknown",
 "summary":"...",
 "line_items":[
   {"panel":"...","operation":"replace|repair","hours_body":1.0,"hours_paint":1.0,"part_cost":0}
 ]
}
""".strip()

def run_ai_estimator(areas, dmg, notes):
    payload={"areas":areas,"damage_types":dmg,"notes":notes}
    c=client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":ESTIMATOR_PROMPT},
            {"role":"user","content":json.dumps(payload)}
        ]
    )
    raw=c.choices[0].message.content or "{}"
    data=safe_json(raw)
    sev=data.get("severity","unknown").lower()
    if sev not in {"minor","moderate","severe"}: sev="unknown"
    summary=data.get("summary","")
    items=data.get("line_items",[])
    if not isinstance(items,list): items=[]
    return {"severity":sev,"summary":summary,"line_items":items}

def price_with_shop(shop: Shop, ai):
    sev=ai["severity"]
    base=shop.pricing.base_floor
    if sev=="minor":
        floor_min,floor_max=base.minor_min,base.minor_max
    elif sev=="moderate":
        floor_min,floor_max=base.moderate_min,base.moderate_max
    elif sev=="severe":
        floor_min,floor_max=base.severe_min,base.severe_max
    else:
        floor_min,floor_max=400,900

    lr=shop.pricing.labor_rates
    mat=shop.pricing.materials_rate

    total_min=0; total_max=0
    for it in ai["line_items"]:
        hb=float(it.get("hours_body") or 0)
        hp=float(it.get("hours_paint") or 0)
        pc=float(it.get("part_cost") or 0)
        labor=hb*lr.body + hp*lr.paint
        materials=hp*mat
        if pc>0:
            total_min+=labor+materials+pc*0.9
            total_max+=labor+materials+pc*1.1
        else:
            total_min+=labor+materials
            total_max+=labor+materials

    total_min=max(total_min, floor_min)
    total_max=max(total_max, floor_max)

    return {
        "severity":sev,
        "summary":ai["summary"],
        "min_cost":int(total_min),
        "max_cost":int(total_max),
        "line_items":ai["line_items"]
    }

def run_estimate(shop, areas, dmg, notes):
    ai=run_ai_estimator(areas, dmg, notes)
    return price_with_shop(shop, ai)


def build_estimate_sms(shop, areas, dmg, est, vin=None):
    L=[]
    L.append(f"AI Damage Estimate for {shop.name}\n")
    if vin and vin.get("vin"):
        L.append(f"Vehicle: {vin.get('year','?')} {vin.get('make','?')} {vin.get('model','?')}\n")
    L.append(f"Severity: {est['severity'].capitalize()}")
    L.append(f"Estimated Cost (Ontario 2025): ${est['min_cost']} – ${est['max_cost']}\n")
    if areas:
        L.append("Areas:\n- "+", ".join(areas)+"\n")
    if dmg:
        L.append("Damage Types:\n- "+", ".join(dmg)+"\n")
    if est["summary"]:
        L.append(est["summary"]+"\n")
    L.append("To book a repair, reply with:")
    L.append("Book Full Name, email@example.com, Nov 29 2pm\n")
    return "\n".join(L)
    # ============================================================
# BOOKING PARSER (ANY ORDER)
# ============================================================

def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None

def extract_name(text: str, email: Optional[str]) -> Optional[str]:
    t = text
    if email: t = t.replace(email, " ")
    t = re.sub(r"\bbook\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " ", t)
    t = re.sub(r"\d{1,2}/\d{1,2}(/\d{2,4})?", " ", t)
    t = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\d{1,2}(:\d{2})?\s*(am|pm)", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    if not t:
        return None
    return " ".join(t.split()[:3])

def parse_booking_any_order(raw: str):
    text = raw.strip()
    email = extract_email(text)
    dt, missing_dt = parse_datetime_any(text)
    name = extract_name(text, email)
    missing=[]
    if not name: missing.append("name")
    if not email: missing.append("email")
    if dt is None: missing.extend(missing_dt)
    missing = list(dict.fromkeys(missing))
    return name, email, dt, missing


# ============================================================
# GOOGLE CALENDAR EVENT (SERVICE ACCOUNT STUB)
# ============================================================

def create_calendar_event(shop: Shop, name: str, email: str, phone: str, dt: datetime, notes: str):
    """
    Your Google service account is shared with the calendar,
    so you CAN plug in actual googleapiclient here later.

    For now we return a stub event so booking never crashes.
    """
    return {"ok": True, "event_id": str(uuid.uuid4())}


# ============================================================
# BOOKING CONFIRMATION MESSAGE
# ============================================================

def build_booking_confirmation(name: str, dt: datetime) -> str:
    return (
        f"Your appointment is confirmed!\n\n"
        f"Name: {name}\n"
        f"Date: {dt.strftime('%Y-%m-%d')}\n"
        f"Time: {dt.strftime('%I:%M %p')}\n\n"
        f"Thank you — we look forward to helping you!"
    )
    # ============================================================
# MAIN TWILIO WEBHOOK
# ============================================================

@app.post("/sms-webhook")
async def sms_webhook(request: Request):
    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""

    token = request.query_params.get("token")
    shop = SHOPS_BY_TOKEN.get(token or "")
    if not shop:
        return PlainTextResponse("Invalid shop token", status_code=400)

    session = get_session(shop, from_number)
    reply = MessagingResponse()

    # =======================================================
    # VIN DETECTION
    # =======================================================
    vin_candidate = body.replace(" ", "").upper()
    if len(vin_candidate) == 17 and vin_candidate.isalnum():
        vin_info = decode_vin(vin_candidate)
        session["vin"] = vin_info
        reply.message(
            f"VIN decoded: {vin_info.get('year','unknown')} {vin_info.get('make','unknown')} {vin_info.get('model','unknown')}.\n\n"
            "Now send 1–3 clear photos of the damage."
        )
        return PlainTextResponse(str(reply), media_type="application/xml")

    # =======================================================
    # PHOTO DETECTION → PRE-SCAN
    # =======================================================
    media_count = int(form.get("NumMedia") or "0")
    if media_count > 0:
        imgs = []
        for i in range(media_count):
            url = form.get(f"MediaUrl{i}")
            ctype = form.get(f"MediaContentType{i}") or "image/jpeg"
            if url:
                try:
                    data = download_media(url)
                    imgs.append(to_data_url(data, ctype))
                except:
                    pass

        if not imgs:
            reply.message("I couldn't read the photos. Please try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        prescan = run_prescan(imgs)
        session["prescan"] = prescan

        L=[]
        L.append(f"AI Pre-Scan for {shop.name}\n")
        if prescan["areas"]:
            L.append("Visible damage areas:")
            L.append("- " + ", ".join(prescan["areas"]) + "\n")
        if prescan["damage_types"]:
            L.append("Damage types:")
            L.append("- " + ", ".join(prescan["damage_types"]) + "\n")
        if prescan["notes"]:
            L.append("Notes:")
            L.append(prescan["notes"] + "\n")
        L.append("If this looks correct, reply 1.")
        L.append("If it's off, reply 2 and send clearer photos.")
        L.append("\nOptional: text your 17-character VIN anytime.")

        reply.message("\n".join(L))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # =======================================================
    # CONFIRM PRE-SCAN
    # =======================================================
    if body == "1" and "prescan" in session:
        ps = session["prescan"]
        areas, dmg, notes = ps["areas"], ps["damage_types"], ps["notes"]

        estimate = run_estimate(shop, areas, dmg, notes)
        session["estimate"] = estimate

        vin = session.get("vin")
        sms = build_estimate_sms(shop, areas, dmg, estimate, vin)
        reply.message(sms)
        return PlainTextResponse(str(reply), media_type="application/xml")

    # =======================================================
    # REJECT PRE-SCAN
    # =======================================================
    if body == "2" and "prescan" in session:
        session.pop("prescan", None)
        reply.message("No problem — please send clearer photos of the damage.")
        return PlainTextResponse(str(reply), media_type="application/xml")

    # =======================================================
    # BOOKING LOGIC — ONLY ALLOWED AFTER ESTIMATE
    # =======================================================
    lower = body.lower()
    looks_booking = False

    if lower.startswith("book"):
        looks_booking = True
    elif extract_email(body):
        if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", lower) or \
           re.search(r"\d{4}-\d{2}-\d{2}", lower) or \
           re.search(r"\d{1,2}/\d{1,2}", lower):
            looks_booking = True

    if looks_booking:
        if "estimate" not in session:
            reply.message(
                "Before booking, I need photos to generate an estimate.\n"
                "Please send 1–3 clear pictures of the damage."
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        name, email, dt, missing = parse_booking_any_order(body)

        if missing:
            reply.message(
                "I couldn't read all booking details.\n\n"
                "Please include:\n"
                "- Full name\n"
                "- Email address\n"
                "- Date & time (e.g. Nov 29 2pm)\n\n"
                "Example:\n"
                "Book John Doe, john@example.com, Nov 29 2pm"
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        if not shop_open(shop, dt):
            reply.message(
                "That time is outside the shop’s hours or unavailable.\n"
                "Please choose another date/time."
            )
            return PlainTextResponse(str(reply), media_type="application/xml")

        event = create_calendar_event(
            shop,
            name=name,
            email=email,
            phone=from_number,
            dt=dt,
            notes="AI estimate booking"
        )

        if not event.get("ok"):
            reply.message("There was a problem creating your booking. Please try again.")
            return PlainTextResponse(str(reply), media_type="application/xml")

        reply.message(build_booking_confirmation(name, dt))
        return PlainTextResponse(str(reply), media_type="application/xml")

    # =======================================================
    # DEFAULT INTRO MESSAGE
    # =======================================================
    intro=[]
    intro.append(f"Hi from {shop.name}!\n")
    intro.append("To get an AI damage estimate:")
    intro.append("1) Send 1–3 photos of the damage.")
    intro.append("2) I'll analyze them with AI.")
    intro.append("3) Confirm with 1.")
    intro.append("4) Then I'll send your full estimate.\n")
    intro.append("Optional:")
    intro.append("- Text your 17-character VIN to decode your vehicle.")
    intro.append("- After your estimate, book with:")
    intro.append("  Book Full Name, email@example.com, Nov 29 2pm")

    reply.message("\n".join(intro))
    return PlainTextResponse(str(reply), media_type="application/xml")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Estimator Running"}
