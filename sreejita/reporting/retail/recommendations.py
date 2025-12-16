# =====================================================
# Retail Recommendations — v2.8.1 (Quantified + Timed)
# =====================================================

def generate_retail_recommendations(df, kpis, insights=None):
    recommendations = []
    insights = insights or []

    total_sales = kpis.get("total_sales", 0)
    avg_discount = kpis.get("average_discount")
    profit_margin = kpis.get("profit_margin", 0)

    # -------------------------------------------------
    # HIGH PRIORITY — Shipping Cost Optimization
    # -------------------------------------------------
    shipping_ratio = kpis.get("shipping_cost_ratio")
    target_shipping = kpis.get("target_shipping_cost_ratio", 0.09)

    if shipping_ratio and shipping_ratio > target_shipping:
        savings = (shipping_ratio - target_shipping) * total_sales

        recommendations.append({
            "action": "Optimize shipping cost structure",
            "priority": "HIGH",
            "expected_impact": f"${savings:,.0f} annual cost savings",
            "timeline": "5–7 days",
            "owner": "Procurement / Operations",
            "success_metric": "Achieve ≤9.0% shipping cost ratio",
            "rationale": (
                f"Current shipping costs are {shipping_ratio*100:.1f}% of sales, "
                f"above the 9.0% target."
            ),
        })

    # -------------------------------------------------
    # MEDIUM PRIORITY — Discount Strategy Optimization
    # -------------------------------------------------
    if avg_discount is not None and avg_discount > 0.12:
        # Conservative and optimistic scenarios
        conservative_gain = total_sales * 0.015
        optimistic_gain = total_sales * 0.02

        recommendations.append({
            "action": "Review pricing and discount strategy",
            "priority": "MEDIUM",
            "expected_impact": (
                f"+${conservative_gain:,.0f} – ${optimistic_gain:,.0f} "
                f"annual profit"
            ),
            "timeline": "2 weeks",
            "owner": "Pricing & Sales Team",
            "success_metric": "Reduce avg discount to 12% without volume loss",
            "rationale": (
                f"Reducing average discount from {avg_discount*100:.1f}% "
                f"to 12% improves margin by ~1.5–2.0%."
            ),
        })

    # -------------------------------------------------
    # FALLBACK
    # -------------------------------------------------
    if not recommendations:
        recommendations.append({
            "action": "Maintain current retail operations",
            "priority": "LOW",
            "expected_impact": "Operational stability",
            "timeline": "Ongoing",
            "owner": "Retail Operations",
            "success_metric": "Maintain KPI performance within thresholds",
            "rationale": "All monitored KPIs are within acceptable ranges.",
        })

    return recommendations
