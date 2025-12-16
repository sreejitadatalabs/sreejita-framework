# =====================================================
# Retail Threshold-Based Insights — v2.7 (FINAL)
# =====================================================

def generate_retail_insights(df, kpis):
    insights = []   # ✅ ALWAYS initialize list

    # -------------------------
    # Shipping Cost Ratio
    # -------------------------
    ratio = kpis.get("shipping_cost_ratio")
    if ratio is not None:
        if ratio <= 0.09:
            level = "GOOD"
            msg = "Shipping costs are within the optimal range."
        elif ratio <= 0.11:
            level = "WARNING"
            msg = "Shipping costs exceed the recommended threshold."
        else:
            level = "RISK"
            msg = "Shipping costs are critically high."

        insights.append({
            "metric": "shipping_cost_ratio",
            "level": level,
            "title": "Shipping Cost Efficiency",
            "value": f"{ratio*100:.1f}%",
            "what": f"Shipping costs account for {ratio*100:.1f}% of total sales.",
            "why": msg,
            "so_what": "Elevated shipping costs directly reduce profit margins.",
        })

    # -------------------------
    # Profit Margin
    # -------------------------
    margin = kpis.get("profit_margin")
    if margin is not None:
        if margin >= 0.12:
            level = "GOOD"
            msg = "Profit margins are healthy."
        elif margin >= 0.09:
            level = "WARNING"
            msg = "Profit margins are under pressure."
        else:
            level = "RISK"
            msg = "Profit margins are critically low."

        insights.append({
            "metric": "profit_margin",
            "level": level,
            "title": "Profitability Health",
            "value": f"{margin*100:.1f}%",
            "what": f"Overall profit margin is {margin*100:.1f}%.",
            "why": msg,
            "so_what": "Low margins limit reinvestment and growth capacity.",
        })

    return insights   # ✅ GUARANTEED LIST
