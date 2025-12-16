# =====================================================
# Retail Recommendations — v2.7 (Threshold-Aware)
# =====================================================

def generate_retail_recommendations(df, kpis, insights=None):
    recommendations = []

    insights = insights or []

    # Normalize insights (backward compatible)
    normalized = []
    for ins in insights:
        if isinstance(ins, dict):
            normalized.append(ins)
        elif isinstance(ins, str):
            normalized.append({"metric": None, "level": "INFO", "text": ins})

    # -------------------------------------------------
    # Shipping Cost Optimization
    # -------------------------------------------------
    for ins in normalized:
        if ins.get("metric") == "shipping_cost_ratio" and ins.get("level") in {"WARNING", "RISK"}:
            ratio = kpis.get("shipping_cost_ratio", 0)
            total_sales = kpis.get("total_sales", 0)

            target = 0.09
            potential_savings = max(ratio - target, 0) * total_sales

            recommendations.append({
                "action": "Optimize shipping cost structure",
                "priority": "HIGH",
                "expected_impact": f"Potential savings of approximately ${potential_savings:,.0f} annually",
                "rationale": ins.get("why", ""),
            })

    # -------------------------------------------------
    # Profit Margin Protection
    # -------------------------------------------------
    if kpis.get("profit_margin", 1) < 0.12:
        recommendations.append({
            "action": "Review pricing and discount strategy",
            "priority": "MEDIUM",
            "expected_impact": "Improved margin stability and profitability",
            "rationale": "Profit margins are below the optimal threshold.",
        })

    # -------------------------------------------------
    # Fallback
    # -------------------------------------------------
    if not recommendations:
        recommendations.append({
            "action": "Maintain current retail operations",
            "priority": "LOW",
            "expected_impact": "Operational stability",
            "rationale": "All monitored KPIs are within acceptable ranges.",
        })

    return recommendations
