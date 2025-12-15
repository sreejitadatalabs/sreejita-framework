import pandas as pd
from typing import Optional


# -------------------------------------------------
# PUBLIC API (v1.x STABLE)
# -------------------------------------------------
def generate_recommendations(
    df: pd.DataFrame,
    sales_col: Optional[str] = None,
    profit_col: Optional[str] = None,
):
    """
    v1.x stable API
    Generates short, executive-level recommendation bullets.
    Compatible with Python 3.9+.
    """

    recommendations = []

    if sales_col and profit_col and sales_col in df.columns and profit_col in df.columns:
        total_sales = df[sales_col].sum()
        total_profit = df[profit_col].sum()
        margin = total_profit / max(total_sales, 1)

        if margin < 0.15:
            recommendations.append(
                "Review pricing and discount strategies to improve profit margins."
            )
        else:
            recommendations.append(
                "Current pricing strategy appears stable with healthy margins."
            )

    if "discount" in df.columns:
        high_discount_rate = (df["discount"] > 0.3).mean()
        if high_discount_rate > 0.2:
            recommendations.append(
                "Reduce the frequency of high-discount transactions to limit margin erosion."
            )

    if df.duplicated().any():
        recommendations.append(
            "Improve data governance to reduce duplicate records."
        )

    if not recommendations:
        recommendations.append(
            "Monitor key performance metrics regularly to detect emerging risks."
        )

    return recommendations


# -------------------------------------------------
# v1.9.9 EXTENSION (SAFE ADDITION)
# -------------------------------------------------
def generate_prescriptive_recommendations(summary_recommendations):
    """
    v1.9.9
    Expands executive recommendations into prescriptive,
    action-oriented decision blocks.
    """

    expanded = []

    archetypes = [
        {
            "priority": "High",
            "outcome": "Reduce downside risk and stabilize performance."
        },
        {
            "priority": "Medium",
            "outcome": "Improve efficiency and unlock incremental gains."
        },
        {
            "priority": "Medium",
            "outcome": "Strengthen governance and decision reliability."
        },
    ]

    for idx, rec in enumerate(summary_recommendations):
        meta = archetypes[idx % len(archetypes)]

        expanded.append({
            "action": rec,
            "rationale": (
                "This action directly addresses an observed pattern in the data "
                "and aligns with best-practice operational controls."
            ),
            "expected_outcome": meta["outcome"],
            "priority": meta["priority"],
        })

    return expanded
