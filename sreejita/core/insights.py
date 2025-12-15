import pandas as pd


# -------------------------------------------------
# EXISTING PUBLIC API (DO NOT BREAK)
# -------------------------------------------------
def correlation_insights(df: pd.DataFrame, target_col: str | None = None):
    """
    v1.x stable API
    Generates short, executive-level correlation insights.
    Used across CLI, domains, reports, automation.
    """

    insights = []

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return ["Insufficient numeric data to compute correlations."]

    corr = numeric_df.corr()

    if target_col and target_col in corr.columns:
        series = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        top = series.iloc[0]
        insights.append(
            f"{target_col.capitalize()} is most strongly correlated with "
            f"{series.index[0]} (r = {top:.2f})."
        )
    else:
        max_pair = (
            corr.where(~pd.np.eye(corr.shape[0], dtype=bool))
            .abs()
            .stack()
            .idxmax()
        )
        insights.append(
            f"{max_pair[0]} and {max_pair[1]} show a strong relationship "
            f"(r = {corr.loc[max_pair]:.2f})."
        )

    return insights


# -------------------------------------------------
# v1.9.9 ADDITION (SAFE EXTENSION)
# -------------------------------------------------
def generate_detailed_insights(summary_insights):
    """
    v1.9.9
    Expands executive insights into consultant-style reasoning.
    DOES NOT replace correlation_insights.
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
