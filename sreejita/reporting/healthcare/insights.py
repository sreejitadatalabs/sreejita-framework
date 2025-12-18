def generate_healthcare_insights(df, kpis):
    insights = []

    outcome = kpis.get("avg_outcome_score", 0)

    if outcome >= 80:
        level = "GOOD"
        msg = "Clinical outcomes are strong."
    elif outcome >= 60:
        level = "WARNING"
        msg = "Outcomes show room for improvement."
    else:
        level = "RISK"
        msg = "Low outcomes indicate quality risks."

    insights.append({
        "metric": "avg_outcome_score",
        "level": level,
        "title": "Clinical Outcomes",
        "value": f"{outcome:.1f}",
        "what": "Average patient outcome score.",
        "why": msg,
        "so_what": "Better outcomes improve patient safety and trust.",
    })

    return insights
