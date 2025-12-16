def retail_insights(df, kpis):
    insights = []

    if kpis["shipping_cost_ratio"] > 0.15:
        insights.append({
            "title": "Shipping costs are eroding margins",
            "evidence": f"Shipping cost ratio is {kpis['shipping_cost_ratio']:.2%}",
            "severity": "high"
        })

    if kpis["discount_rate"] > 0.2:
        insights.append({
            "title": "High discounting may impact profitability",
            "evidence": f"Average discount rate is {kpis['discount_rate']:.2%}",
            "severity": "medium"
        })

    return insights
