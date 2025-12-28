def enforce_recommendations(insights, recommendations):
    """
    Ensures every report has actionable next steps.
    """

    if recommendations:
        return recommendations

    if insights:
        return [{
            "action": "Assign ownership and review high-risk indicators weekly.",
            "priority": "MEDIUM"
        }]

    return [{
        "action": "Continue monitoring KPIs against benchmarks.",
        "priority": "LOW"
    }]
