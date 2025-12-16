def generate_retail_insights(df, kpis):
    insights = []

    # Shipping cost insight
    if "shipping_cost_ratio" in kpis:
        ratio = kpis["shipping_cost_ratio"]

        if ratio > 0.15:
            insights.append({
                "title": "Shipping costs are eroding margins",
                "severity": "high",
                "evidence": f"Shipping cost ratio is {ratio:.2%}",
                "metric": "shipping_cost_ratio"
            })

    # Discount insight
    if "average_discount" in kpis:
        disc = kpis["average_discount"]

        if disc > 0.20:
            insights.append({
                "title": "High discounting may impact profitability",
                "severity": "medium",
                "evidence": f"Average discount rate is {disc:.2%}",
                "metric": "average_discount"
            })

    # Fallback insight (important)
    if not insights:
        insights.append({
            "title": "No critical operational risks detected",
            "severity": "low",
            "evidence": "All monitored retail KPIs are within expected ranges",
            "metric": "overall_health"
        })

    return insights
