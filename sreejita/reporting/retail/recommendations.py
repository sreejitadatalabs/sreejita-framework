def generate_retail_recommendations(insights):
    recommendations = []

    for ins in insights:
        if ins["metric"] == "shipping_cost_ratio":
            recommendations.append({
                "action": "Review shipping contracts and optimize delivery zones",
                "priority": "High",
                "expected_impact": "Margin improvement"
            })

        elif ins["metric"] == "average_discount":
            recommendations.append({
                "action": "Reduce blanket discounting and target high-value segments",
                "priority": "Medium",
                "expected_impact": "Profit stabilization"
            })

    if not recommendations:
        recommendations.append({
            "action": "Maintain current retail operations",
            "priority": "Low",
            "expected_impact": "Operational stability"
        })

    return recommendations
