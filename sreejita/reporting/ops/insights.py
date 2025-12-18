def generate_ops_insights(df, kpis):
    rate = kpis.get("on_time_rate", 0)

    level = "GOOD" if rate > 0.9 else "WARNING" if rate > 0.75 else "RISK"

    return [{
        "metric": "on_time_rate",
        "level": level,
        "title": "Delivery Reliability",
        "value": f"{rate:.1%}",
        "what": "Measures operational timeliness.",
        "why": "Delays reduce efficiency and trust.",
        "so_what": "Improving reliability lowers costs.",
    }]
