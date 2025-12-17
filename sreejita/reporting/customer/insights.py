def generate_customer_insights(df, kpis):
    insights = []

    avg_value = kpis.get("average_customer_value", 0)

    if avg_value >= 500:
        level = "GOOD"
        msg = "Customers generate strong average value."
    elif avg_value >= 200:
        level = "WARNING"
        msg = "Customer value is moderate."
    else:
        level = "RISK"
        msg = "Customer value is low."

    insights.append({
        "metric": "average_customer_value",
        "level": level,
        "title": "Customer Value Health",
        "value": f"${avg_value:,.0f}",
        "what": f"Average customer value is ${avg_value:,.0f}.",
        "why": msg,
        "so_what": "Customer value directly impacts revenue scalability.",
    })

    return insights
