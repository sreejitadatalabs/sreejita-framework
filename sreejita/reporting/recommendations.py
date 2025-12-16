def retail_recommendations(insights):
    recs = []

    for ins in insights:
        if "Shipping costs" in ins["title"]:
            recs.append({
                "action": "Review shipping contracts and zone pricing",
                "priority": "High",
                "expected_impact": "Margin improvement"
            })

    return recs
