def generate_healthcare_recommendations(insights):
    return [{
        "action": "Review discharge and follow-up protocols",
        "priority": "High",
        "expected_impact": "Reduced readmissions"
    }] if insights else []
