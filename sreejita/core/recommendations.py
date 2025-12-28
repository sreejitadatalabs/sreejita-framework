import pandas as pd
from typing import Optional, List, Dict, Any


# -------------------------------------------------
# PUBLIC API (v1.x STABLE)
# -------------------------------------------------
def generate_recommendations(
    df: pd.DataFrame,
    sales_col: Optional[str] = None,
    profit_col: Optional[str] = None,
) -> List[str]:
    """
    v1.x stable API.

    Generates short, executive-level recommendation bullets.
    This function is heuristic-driven, deterministic, and domain-agnostic.

    Notes:
    - Benchmarks are illustrative defaults, not industry guarantees.
    - Column presence is validated defensively.
    """

    recommendations: List[str] = []

    # -----------------------------
    # Profitability Check
    # -----------------------------
    if (
        sales_col
        and profit_col
        and sales_col in df.columns
        and profit_col in df.columns
    ):
        total_sales = df[sales_col].sum()
        total_profit = df[profit_col].sum()

        # Guard against divide-by-zero and malformed data
        margin = total_profit / max(float(total_sales), 1.0)

        if margin < 0.15:
            recommendations.append(
                "Review pricing and discount strategies to improve profit margins."
            )
        else:
            recommendations.append(
                "Current pricing strategy appears stable with healthy margins."
            )

    # -----------------------------
    # Discount Heuristic
    # -----------------------------
    # NOTE: 'discount' is treated as a common retail/ecommerce convention.
    if "discount" in df.columns and pd.api.types.is_numeric_dtype(df["discount"]):
        high_discount_rate = (df["discount"] > 0.30).mean()
        if high_discount_rate > 0.20:
            recommendations.append(
                "Reduce the frequency of deep-discount transactions to limit margin erosion."
            )

    # -----------------------------
    # Data Governance Signal
    # -----------------------------
    if df.duplicated().any():
        recommendations.append(
            "Strengthen data governance controls to eliminate duplicate records."
        )

    # -----------------------------
    # Fallback
    # -----------------------------
    if not recommendations:
        recommendations.append(
            "Continue monitoring key performance indicators to detect emerging risks early."
        )

    return recommendations


# -------------------------------------------------
# v1.9.9 EXTENSION (SAFE ADDITION)
# -------------------------------------------------
def generate_prescriptive_recommendations(
    summary_recommendations: List[str]
) -> List[Dict[str, Any]]:
    """
    v1.9.9 extension.

    Expands executive recommendations into prescriptive,
    action-oriented decision blocks.

    Design intent:
    - Deterministic
    - Domain-agnostic
    - Suitable for executive reports and narratives
    """

    expanded: List[Dict[str, Any]] = []

    archetypes = [
        {
            "priority": "HIGH",
            "outcome": "Reduce downside risk and stabilize performance."
        },
        {
            "priority": "MEDIUM",
            "outcome": "Improve efficiency and unlock incremental gains."
        },
        {
            "priority": "MEDIUM",
            "outcome": "Strengthen governance and decision reliability."
        },
    ]

    for idx, rec in enumerate(summary_recommendations):
        meta = archetypes[idx % len(archetypes)]

        expanded.append({
            "action": rec,
            "rationale": (
                "This action addresses a consistent pattern observed in the data "
                "and aligns with established operational best practices."
            ),
            "expected_outcome": meta["outcome"],
            "priority": meta["priority"],
        })

    return expanded
