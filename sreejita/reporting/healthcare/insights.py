def generate_healthcare_insights(df, kpis):
    score = kpis.get("outcome_score", 0)

    level = "GOOD" if score > 80 else "WARNING" if score > 60 else "RISK"

    return [{
        "metric": "outcome_score",
        "level": level,
        "title": "Clinical Outcomes",
        "value": f"{score:.1f}",
        "what": "Average patient outcome score.",
        "why": "Outcomes reflect care quality.",
        "so_what": "Higher outcomes improve trust and compliance.",
    }]
