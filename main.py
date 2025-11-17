async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig) -> dict:
    """Generate high-quality damage estimates using gpt-4o-mini vision."""
    if not OPENAI_API_KEY:
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.5,
            "vin_used": False,
        }

    model_name = "gpt-4o-mini"

    system_prompt = """
You are a certified Ontario auto-body damage estimator (2025). 
Analyze the images and output JSON ONLY with:

severity: Minor | Moderate | Severe
damage_areas: list of panels specifically damaged
damage_types: dents, scuffs, cracks, sharp dents, etc
recommended_repairs: realistic repair actions
min_cost: CAD
max_cost: CAD
confidence: 0.0–1.0
vin_used: true/false
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build message content
    user_content = [{"type": "text", "text": "Analyze these car damage photos."}]
    if vin:
        user_content.append({"type": "text", "text": f"VIN: {vin}"})

    for url in image_urls[:3]:  # Limit to 3 images for reliability
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 500
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        resp.raise_for_status()
        raw_json = resp.json()["choices"][0]["message"]["content"]
        result = json.loads(raw_json)

        # Sanity defaults
        result.setdefault("severity", "Moderate")
        result.setdefault("damage_areas", [])
        result.setdefault("damage_types", [])
        result.setdefault("recommended_repairs", [])
        result.setdefault("min_cost", 600)
        result.setdefault("max_cost", 1500)
        result.setdefault("confidence", 0.7)
        result.setdefault("vin_used", bool(vin))

        return result

    except Exception as e:
        print("AI ERROR:", e)
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.4,
            "vin_used": bool(vin),
        }
