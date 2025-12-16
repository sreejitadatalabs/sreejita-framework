def generate_healthcare_insights(df, kpis):
    insights = []

    if kpis["readmission_rate"] > 0.15:
        insights.append({
            "title": "High patient readmission rate",
            "severity": "high",
            "evidence": f"Readmission rate at {kpis['readmission_rate']:.1%}",
            "metric": "readmission_rate"
        })

    return insights
