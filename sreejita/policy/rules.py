def check_min_confidence(decision, min_confidence: float):
    if decision.confidence < min_confidence:
        return {
            "status": "allowed_with_warning",
            "reason": f"confidence_below_{min_confidence}"
        }
    return None


def block_if_unknown_domain(decision):
    if decision.selected_domain == "unknown":
        return {
            "status": "blocked",
            "reason": "unknown_domain_detected"
        }
    return None
