# =====================================================
# Retail Threshold-Based Insights — v2.8
# =====================================================

def generate_retail_insights(df, kpis):
    insights = []

    # -------------------------
    # 1. Shipping Cost Ratio
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
    # 2. Profit Margin Health
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
            "so_what": "Margin pressure limits reinvestment and growth capacity.",
        })

    # -------------------------
    # 3. Category Concentration
    # -------------------------
    if "category" in df.columns and "sales" in df.columns:
        totals = df.groupby("category")["sales"].sum()
        top_share = totals.max() / totals.sum()

        if top_share <= 0.40:
            level = "GOOD"
            msg = "Revenue is well diversified across categories."
        elif top_share <= 0.55:
            level = "WARNING"
            msg = "Revenue shows moderate concentration risk."
        else:
            level = "RISK"
            msg = "Revenue is highly dependent on a single category."

        insights.append({
            "metric": "category_concentration",
            "level": level,
            "title": "Category Revenue Concentration",
            "value": f"{top_share*100:.1f}%",
            "what": f"The top category contributes {top_share*100:.1f}% of total sales.",
            "why": msg,
            "so_what": "High concentration increases exposure to category-specific shocks.",
        })

    return insights
