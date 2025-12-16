def generate_finance_insights(df, kpis):
    insights = []

    if kpis["expense_ratio"] > 0.6:
        insights.append({
            "title": "Expenses consuming majority of revenue",
            "severity": "high",
            "evidence": f"Expense ratio at {kpis['expense_ratio']:.1%}",
            "metric": "expense_ratio"
        })

    return insights
