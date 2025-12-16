def generate_ops_insights(df, kpis):
    insights = []

    if kpis["sla_breach_rate"] > 0.1:
        insights.append({
            "title": "Frequent SLA breaches detected",
            "severity": "high",
            "evidence": f"{kpis['sla_breach_rate']:.1%} processes breached SLA",
            "metric": "sla_breach_rate"
        })

    return insights
