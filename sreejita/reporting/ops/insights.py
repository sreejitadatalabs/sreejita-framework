def generate_ops_insights(df, kpis):
    insights = []

    on_time_rate = kpis.get("on_time_rate", 0)

    if on_time_rate >= 0.9:
        level = "GOOD"
        msg = "Operational reliability is strong."
    elif on_time_rate >= 0.75:
        level = "WARNING"
        msg = "Delivery delays are emerging."
    else:
        level = "RISK"
        msg = "Operational delays are frequent."

    insights.append({
        "metric": "on_time_rate",
        "level": level,
        "title": "Delivery Reliability",
        "value": f"{on_time_rate:.1%}",
        "what": "Measures the percentage of on-time operations.",
        "why": msg,
        "so_what": "Delays increase costs and reduce customer trust.",
    })

    return insights
