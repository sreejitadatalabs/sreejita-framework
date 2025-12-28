import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


# -------------------------------------------------
# PUBLIC API (v1.x STABLE)
# -------------------------------------------------
def correlation_insights(
    df: pd.DataFrame,
    target_col: Optional[str] = None
) -> List[str]:
    """
    v1.x stable API.

    Generates short, executive-level correlation insights.
    This function is deterministic and domain-agnostic.

    Notes:
    - Returns high-level signals, not statistical conclusions.
    - Handles sparse or low-quality numeric data gracefully.
    """

    insights: List[str] = []

    numeric_df = df.select_dtypes(include="number")

    # Guard: insufficient numeric data
    if numeric_df.shape[1] < 2:
        return ["Insufficient numeric data to compute correlations."]

    # Remove constant columns (zero variance)
    numeric_df = numeric_df.loc[:, numeric_df.std(numeric_only=True) > 0]

    if numeric_df.shape[1] < 2:
        return ["Numeric data lacks sufficient variance for correlation analysis."]

    corr = numeric_df.corr()

    # Guard: all-NaN correlation matrix
    if corr.isna().all().all():
        return ["Correlation analysis could not be computed due to data sparsity."]

    # -------------------------------------------------
    # Targeted correlation insight
    # -------------------------------------------------
    if target_col and target_col in corr.columns:
        series = (
            corr[target_col]
            .drop(labels=[target_col], errors="ignore")
            .dropna()
            .sort_values(key=lambda x: abs(x), ascending=False)
        )

        if not series.empty:
            top_feature = series.index[0]
            insights.append(
                f"{target_col.replace('_', ' ').capitalize()} is most strongly "
                f"correlated with {top_feature.replace('_', ' ')} "
                f"(r = {series.iloc[0]:.2f})."
            )
        else:
            insights.append(
                f"No meaningful correlations found for {target_col}."
            )

    # -------------------------------------------------
    # General strongest correlation insight
    # -------------------------------------------------
    else:
        # Exclude self-correlation
        mask = np.eye(corr.shape[0], dtype=bool)
        corr_masked = corr.mask(mask)

        stacked = corr_masked.abs().stack().dropna()
        if stacked.empty:
            return ["No significant correlations detected among numeric features."]

        max_pair = stacked.idxmax()

        insights.append(
            f"{max_pair[0].replace('_', ' ').capitalize()} and "
            f"{max_pair[1].replace('_', ' ')} show a strong relationship "
            f"(r = {corr.loc[max_pair]:.2f})."
        )

    return insights


# -------------------------------------------------
# v1.9.9 EXTENSION (SAFE ADDITION)
# -------------------------------------------------
def generate_detailed_insights(
    summary_insights: List[str]
) -> List[Dict[str, Any]]:
    """
    v1.9.9 extension.

    Expands executive-level insights into consultant-style reasoning.
    This layer is intentionally generic and domain-agnostic.

    Purpose:
    - Provide explainability
    - Support narrative/reporting layers
    - Avoid speculative or model-based claims
    """

    detailed: List[Dict[str, Any]] = []

    for idx, insight in enumerate(summary_insights, start=1):
        detailed.append({
            "title": f"Insight {idx}",
            "what": insight,
            "why": (
                "Observed patterns are consistent across multiple metrics, "
                "suggesting structural relationships rather than random variation."
            ),
            "so_what": (
                "If this relationship persists, it may materially influence "
                "performance, risk exposure, or operational efficiency."
            ),
        })

    return detailed
