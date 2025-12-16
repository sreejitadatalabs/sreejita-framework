def generate_finance_recommendations(insights):
    return [{
        "action": "Audit high-cost expense categories",
        "priority": "High",
        "expected_impact": "Margin improvement"
    }] if insights else []
