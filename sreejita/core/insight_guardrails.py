# Prevents "INFO / Stable" when risk KPIs are breached

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "RISK": 2, "CRITICAL": 3}

def enforce_insight_guardrails(insights, kpis, thresholds):
    """
    Ensures insight severity matches KPI reality.
    """

    max_required_severity = 0

    for kpi, rules in thresholds.items():
        value = kpis.get(kpi)
        if value is None:
            continue

        if "critical" in rules and value >= rules["critical"]:
            max_required_severity = max(max_required_severity, SEVERITY_ORDER["CRITICAL"])
        elif "warning" in rules and value >= rules["warning"]:
            max_required_severity = max(max_required_severity, SEVERITY_ORDER["WARNING"])

    # Remove invalid INFO insights
    cleaned = []
    for ins in insights:
        if SEVERITY_ORDER.get(ins["level"], 0) >= max_required_severity:
            cleaned.append(ins)

    # Force at least one serious insight if needed
    if max_required_severity > 0 and not cleaned:
        cleaned.append({
            "level": "WARNING",
            "title": "Operational Risk Detected",
            "so_what": "Key metrics exceed safe operating thresholds."
        })

    return cleaned
