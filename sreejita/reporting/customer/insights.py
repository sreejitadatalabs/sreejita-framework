def generate_customer_insights(df, kpis):
    insights = []

    if kpis["churn_proxy_rate"] > 0.3:
        insights.append({
            "title": "High customer inactivity risk",
            "severity": "high",
            "evidence": f"{kpis['churn_proxy_rate']:.1%} customers inactive > 6 months",
            "metric": "churn_proxy_rate"
        })

    if not insights:
        insights.append({
            "title": "Customer engagement is stable",
            "severity": "low",
            "evidence": "No major inactivity patterns detected",
            "metric": "engagement_health"
        })

    return insights
