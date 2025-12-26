# severity_engine.py

from typing import Dict, Any


def calculate_severity(visual_flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    visual_flags example:
    {
        "wheel_displacement": True,
        "asymmetrical_impact": True,
        "ride_height_anomaly": False,
        "debris_field_large": True
    }
    """

    severity_score = 0
    triggered_rules = []

    # Rule A — Wheel & Suspension
    if visual_flags.get("wheel_displacement"):
        severity_score += 3
        triggered_rules.append("Wheel displacement detected")

    # Rule B — Asymmetrical Impact
    if visual_flags.get("asymmetrical_impact"):
        severity_score += 2
        triggered_rules.append("Asymmetrical impact detected")

    # Rule C — Ride Height
    if visual_flags.get("ride_height_anomaly"):
        severity_score += 2
        triggered_rules.append("Ride height anomaly detected")

    # Rule D — Debris Field
    if visual_flags.get("debris_field_large"):
        severity_score += 1
        triggered_rules.append("Large debris field detected")

    # Severity Ladder
    if severity_score >= 5:
        severity_level = "Structural Risk"
        labor_hours = (16, 28)
        confidence = "Low–Medium"
        mandatory_ops = [
            "Suspension inspection",
            "Wheel alignment",
            "Structural measurement",
            "Pre-scan diagnostics",
            "Post-scan diagnostics",
        ]

    elif severity_score >= 3:
        severity_level = "Panel + Mechanical Risk"
        labor_hours = (12, 20)
        confidence = "Medium"
        mandatory_ops = [
            "Wheel alignment",
            "Pre-scan diagnostics",
            "Post-scan diagnostics",
        ]

    else:
        severity_level = "Cosmetic / Panel"
        labor_hours = (4, 10)
        confidence = "High"
        mandatory_ops = []

    return {
        "severity_level": severity_level,
        "labor_hours": labor_hours,
        "confidence": confidence,
        "mandatory_operations": mandatory_ops,
        "triggered_rules": triggered_rules,
    }
