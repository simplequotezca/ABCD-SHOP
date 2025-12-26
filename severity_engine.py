# severity_engine.py
from typing import Dict, Any


def infer_visual_flags(ai: Dict[str, Any]) -> Dict[str, bool]:
    """
    Convert AI vision output into deterministic escalation flags.
    Conservative by design: better to under-trigger than over-trigger.
    """

    areas = " ".join(ai.get("damaged_areas", [])).lower()
    ops = " ".join(ai.get("operations", [])).lower()
    notes = (ai.get("notes") or "").lower()

    text = " ".join([areas, ops, notes])

    return {
        # Wheel / suspension involvement
        "wheel_displacement": any(k in text for k in [
            "wheel",
            "suspension",
            "control arm",
            "tie rod",
            "alignment",
            "bent wheel",
        ]),

        # One-sided or offset impacts
        "asymmetrical_impact": any(k in text for k in [
            "one side",
            "left side",
            "right side",
            "offset impact",
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

    if flags.get("wheel_displacement"):
        score += 3
        reasons.append("Wheel / suspension displacement")

    if flags.get("frame_signal"):
        score += 3
        reasons.append("Structural / frame involvement")

    if flags.get("asymmetrical_impact"):
        score += 2
        reasons.append("Asymmetrical impact")

    if flags.get("ride_height_anomaly"):
        score += 2
        reasons.append("Ride height anomaly")

    if flags.get("debris_field_large"):
        score += 1
        reasons.append("Large debris field")

    # --------------------------------------------------------
    # HARD FLOOR RULE
    # Asymmetry combined with any other signal is never cosmetic
    # --------------------------------------------------------
    if flags.get("asymmetrical_impact") and score >= 3:
        return {
            "severity": "Panel + Mechanical Risk",
            "confidence": "Medium",
            "labor_range": (12, 20),
            "reasons": reasons,
        }

    # -------------------
    # SEVERITY LADDER
    # -------------------
    if score >= 5:
        return {
            "severity": "Structural Risk",
            "confidence": "Medium",
            "labor_range": (16, 28),
            "reasons": reasons,
        }

    if score >= 3:
        return {
            "severity": "Panel + Mechanical Risk",
            "confidence": "Medium",
            "labor_range": (12, 20),
            "reasons": reasons,
        }

    return {
        "severity": "Cosmetic / Panel",
        "confidence": "High",
        "labor_range": (4, 10),
        "reasons": reasons,
    }
