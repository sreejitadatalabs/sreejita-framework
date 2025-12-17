def generate_customer_recommendations(df, kpis, insights=None):
    recommendations = []

    avg_value = kpis.get("average_customer_value", 0)

    if avg_value < 200:
        recommendations.append({
            "action": "Improve customer retention and upsell programs",
            "priority": "HIGH",
            "expected_impact": "+$150,000 – $300,000 annual revenue",
            "timeline": "2–4 weeks",
            "owner": "Customer Success / Marketing",
            "success_metric": "Increase average customer value above $300",
            "rationale": "Low customer value limits revenue growth.",
        })

    if not recommendations:
        recommendations.append({
            "action": "Maintain current customer engagement strategy",
            "priority": "LOW",
            "expected_impact": "Stable customer revenue",
            "timeline": "Ongoing",
            "owner": "Customer Operations",
            "success_metric": "Sustain customer value KPIs",
            "rationale": "Customer metrics are within healthy thresholds.",
        })

    return recommendations
