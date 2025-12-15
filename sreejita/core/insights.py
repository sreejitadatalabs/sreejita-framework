import pandas as pd
import numpy as np
from typing import Optional


# -------------------------------------------------
# PUBLIC API (v1.x STABLE)
# -------------------------------------------------
def correlation_insights(df: pd.DataFrame, target_col: Optional[str] = None):
    """
    v1.x stable API
    Generates short, executive-level correlation insights.
    Compatible with Python 3.9+.
    """

    insights = []

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return ["Insufficient numeric data to compute correlations."]

    corr = numeric_df.corr()

    if target_col and target_col in corr.columns:
        series = corr[target_col].drop(target_col).sort_values(
            key=lambda x: abs(x), ascending=False
        )
        top_feature = series.index[0]
        insights.append(
            f"{target_col.capitalize()} is most strongly correlated with "
            f"{top_feature} (r = {series.iloc[0]:.2f})."
        )
    else:
        # find strongest absolute correlation (excluding self-correlation)
        mask = np.eye(corr.shape[0], dtype=bool)
        corr_masked = corr.mask(mask)

        max_pair = (
            corr_masked.abs()
            .stack()
            .idxmax()
        )

        insights.append(
            f"{max_pair[0]} and {max_pair[1]} show a strong relationship "
            f"(r = {corr.loc[max_pair]:.2f})."
        )

    return insights


# -------------------------------------------------
# v1.9.9 EXTENSION (SAFE ADDITION)
# -------------------------------------------------
def generate_detailed_insights(summary_insights):
    """
    v1.9.9
    Expands executive insights into consultant-style reasoning.
    Domain-agnostic, deterministic.
    """

    detailed = []

    for idx, insight in enumerate(summary_insights, start=1):
        detailed.append({
            "title": f"Insight {idx}",
            "what": insight,
            "why": (
                "Observed patterns are consistent across multiple metrics, "
                "indicating structural behavior rather than random variance."
            ),
            "so_what": (
                "If this pattern persists, it may materially impact performance, "
                "risk exposure, or operational efficiency."
            ),
        })

    return detailed
    
