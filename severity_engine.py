# severity_engine.py
from typing import Dict, Any


def infer_visual_flags(ai: Dict[str, Any]) -> Dict[str, bool]:
    """
    Convert AI vision output into deterministic escalation flags.
    Rule-based and conservative. No hallucination.
    """

    areas = " ".join(ai.get("damaged_areas", [])).lower()
    ops = " ".join(ai.get("operations", [])).lower()
    notes = (ai.get("notes") or "").lower()

    text = " ".join([areas, ops, notes])

    return {
        # Wheel / suspension involvement
        "wheel_displacement": any(k in text for k in [
            "wheel",
            "tire",
            "rim",
            "suspension",
            "control arm",
            "tie rod",
            "alignment",
            "bent wheel",
        ]),

        # Front corner / offset collision pattern
        "front_corner_impact": (
            ("bumper" in text) and
            (("fender" in text) or ("headlight" in text))
        ),

        # One-sided or offset impacts
        "asymmetrical_impact": any(k in text for k in [
            "one side",
            "left side",
            "right side",
            "offset",
        ]),

        # Vehicle stance issues
        "ride_height_anomaly": any(k in text for k in [
            "lean",
            "uneven",
            "lower on one side",
        ]),

        # Force / energy indicators
        "debris_field_large": any(k in text for k in [
            "debris",
            "scattered",
            "fragments",
        ]),

        # Structural / frame language
        "frame_signal": any(k in text for k in [
            "frame",
            "unibody",
            "rail",
            "apron",
            "core support",
            "structural",
        ]),
    }


def calculate_severity(flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    Hard severity ladder.
    Once escalated, severity cannot be downgraded.
    """

    score = 0
    reasons = []

    # --------------------
    # PRIMARY ESCALATORS
    # --------------------
    if flags.get("wheel_displacement"):
        score += 3
        reasons.append("Wheel / suspension involvement")

    if flags.get("frame_signal"):
        score += 4
        reasons.append("Structural / frame involvement")

    if flags.get("front_corner_impact"):
        score += 2
        reasons.append("Front corner collision pattern")

    if flags.get("asymmetrical_impact"):
        score += 2
        reasons.append("Asymmetrical / offset impact")

    if flags.get("ride_height_anomaly"):
        score += 2
        reasons.append("Ride height anomaly")

    if flags.get("debris_field_large"):
        score += 1
        reasons.append("High-energy debris field")

    # --------------------------------------------------------
    # HARD FLOORS (NON-NEGOTIABLE)
    # --------------------------------------------------------

    # Any front-corner impact is NEVER cosmetic
    if flags.get("front_corner_impact"):
        return {
            "severity": "Panel + Mechanical Risk",
            "confidence": "Medium",
            "labor_range": (14, 26),
            "reasons": reasons,
        }

    # Any wheel or frame signal is never cosmetic
    if flags.get("wheel_displacement") or flags.get("frame_signal"):
        return {
            "severity": "Structural Risk" if flags.get("frame_signal") else "Panel + Mechanical Risk",
            "confidence": "Medium",
            "labor_range": (20, 36) if flags.get("frame_signal") else (14, 26),
            "reasons": reasons,
        }

    # -------------------
    # SCORE-BASED LADDER
    # -------------------
    if score >= 5:
        return {
            "severity": "Structural Risk",
            "confidence": "Medium",
            "labor_range": (20, 36),
            "reasons": reasons,
        }

    if score >= 3:
        return {
            "severity": "Panel + Mechanical Risk",
            "confidence": "Medium",
            "labor_range": (14, 26),
            "reasons": reasons,
        }

    return {
        "severity": "Cosmetic / Panel",
        "confidence": "High",
        "labor_range": (6, 14),
        "reasons": reasons,
    }
