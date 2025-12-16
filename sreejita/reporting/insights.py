def generate_retail_insights(df, kpis):
    insights = []

    if "shipping_cost_ratio" in kpis and kpis["shipping_cost_ratio"] > 0.15:
        insights.append({
            "title": "Shipping costs are eroding margins",
            "severity": "high",
            "evidence": f"Shipping cost ratio is {kpis['shipping_cost_ratio']:.2%}",
            "metric": "shipping_cost_ratio"
        })

    if "average_discount" in kpis and kpis["average_discount"] > 0.20:
        insights.append({
            "title": "High discounting may impact profitability",
            "severity": "medium",
            "evidence": f"Average discount rate is {kpis['average_discount']:.2%}",
            "metric": "average_discount"
        })

    if not insights:
        insights.append({
            "title": "No critical operational risks detected",
            "severity": "low",
            "evidence": "All monitored KPIs are within expected ranges",
            "metric": "overall_health"
        })

    return insights
