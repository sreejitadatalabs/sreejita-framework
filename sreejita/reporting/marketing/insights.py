def generate_marketing_insights(df, kpis):
    insights = []

    conv = kpis.get("conversion_rate", 0)

    if conv >= 0.1:
        level = "GOOD"
        msg = "Campaigns are converting effectively."
    elif conv >= 0.05:
        level = "WARNING"
        msg = "Conversion rates are moderate."
    else:
        level = "RISK"
        msg = "Low conversion rates indicate inefficient spend."

    insights.append({
        "metric": "conversion_rate",
        "level": level,
        "title": "Campaign Conversion Performance",
        "value": f"{conv:.1%}",
        "what": "Percentage of users who converted.",
        "why": msg,
        "so_what": "Low conversion increases cost per acquisition.",
    })

    return insights
