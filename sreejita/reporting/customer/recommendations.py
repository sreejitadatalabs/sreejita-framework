def generate_customer_recommendations(insights):
    recs = []

    for ins in insights:
        if ins["metric"] == "churn_proxy_rate":
            recs.append({
                "action": "Launch re-engagement campaigns for inactive customers",
                "priority": "High",
                "expected_impact": "Retention improvement"
            })

    return recs
