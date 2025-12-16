from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


# =====================================================
# Retail Threshold-Based Insights (v2.7)
# =====================================================

def generate_retail_insights(df, kpis):
    insights = []

    # -------------------------
    # 1. Shipping Cost Ratio
    # -------------------------
    shipping_ratio = kpis.get("shipping_cost_ratio")

    if shipping_ratio is not None:
        if shipping_ratio <= 0.09:
            level = "GOOD"
            msg = "Shipping costs are well controlled."
        elif shipping_ratio <= 0.11:
            level = "WARNING"
            msg = "Shipping costs are slightly above the optimal range."
        else:
            level = "RISK"
            msg = "Shipping costs are materially impacting margins."

        insights.append({
            "metric": "shipping_cost_ratio",
            "level": level,
            "value": round(shipping_ratio * 100, 2),
            "title": "Shipping Cost Efficiency",
            "what": f"Shipping costs account for {shipping_ratio:.1%} of total sales.",
            "why": msg,
            "so_what": "High shipping costs directly reduce profitability.",
        })

    # -------------------------
    # 2. Profit Margin
    # -------------------------
    profit_margin = kpis.get("profit_margin")

    if profit_margin is not None:
        if profit_margin >= 0.12:
            level = "GOOD"
            msg = "Profit margins are healthy."
        elif profit_margin >= 0.09:
            level = "WARNING"
            msg = "Profit margins are under mild pressure."
        else:
            level = "RISK"
            msg = "Profit margins are critically low."

        insights.append({
            "metric": "profit_margin",
            "level": level,
            "value": round(profit_margin * 100, 2),
            "title": "Profitability Health",
            "what": f"Overall profit margin is {profit_margin:.1%}.",
            "why": msg,
            "so_what": "Sustained margin pressure limits reinvestment capacity.",
        })

    # -------------------------
    # 3. Average Order Value (AOV)
    # -------------------------
    aov = kpis.get("average_order_value")
    target_aov = kpis.get("target_aov", aov)

    if aov and target_aov:
        delta_pct = (aov - target_aov) / target_aov

        if delta_pct >= 0:
            level = "GOOD"
            msg = "Average order value meets or exceeds expectations."
        elif delta_pct >= -0.05:
            level = "WARNING"
            msg = "Average order value is slightly below target."
        else:
            level = "RISK"
            msg = "Average order value is significantly below target."

        insights.append({
            "metric": "average_order_value",
            "level": level,
            "value": round(aov, 2),
            "title": "Order Value Performance",
            "what": f"Average order value is ${aov:,.2f}.",
            "why": msg,
            "so_what": "Lower order values reduce revenue leverage per transaction.",
        })

    # -------------------------
    # 4. Category Concentration
    # -------------------------
    if "category" in df.columns and "sales" in df.columns:
        cat_sales = (
            df.groupby("category")["sales"].sum()
            .sort_values(ascending=False)
        )

        top_share = cat_sales.iloc[0] / cat_sales.sum()

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
            "value": round(top_share * 100, 2),
            "title": "Category Revenue Concentration",
            "what": f"Top category contributes {top_share:.1%} of total sales.",
            "why": msg,
            "so_what": "High concentration increases exposure to category-specific shocks.",
        })

    return insights

