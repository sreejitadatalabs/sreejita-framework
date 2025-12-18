def generate_finance_insights(df, kpis):
    insights = []

    margin = kpis.get("profit_margin", 0)

    if margin >= 0.25:
        level = "GOOD"
        msg = "Profit margins are strong."
    elif margin >= 0.1:
        level = "WARNING"
        msg = "Margins are tightening."
    else:
        level = "RISK"
        msg = "Low margins threaten sustainability."

    insights.append({
        "metric": "profit_margin",
        "level": level,
        "title": "Profitability Health",
        "value": f"{margin:.1%}",
        "what": "Measures profitability relative to revenue.",
        "why": msg,
        "so_what": "Margins directly affect long-term financial stability.",
    })

    return insights
