from typing import Dict, List


def profile_schema(
    df,
    expected_columns: List[str],
    computed_kpis: Dict[str, float],
) -> Dict:
    """
    Profiles dataset schema and KPI coverage.
    Safe, non-blocking, informational only.
    """

    present_cols = set(df.columns)
    expected_cols = set(expected_columns)

    missing_columns = sorted(list(expected_cols - present_cols))
    extra_columns = sorted(list(present_cols - expected_cols))

    # KPI coverage
    total_kpis = len(computed_kpis)
    computed_count = sum(1 for v in computed_kpis.values() if v is not None)
    coverage = round(computed_count / total_kpis, 2) if total_kpis else 1.0

    # Null density (optional but useful)
    null_ratio = (
        df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if df.shape[0] and df.shape[1]
        else 0
    )

    return {
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "kpi_coverage": coverage,
        "computed_kpis": computed_count,
        "total_kpis": total_kpis,
        "null_ratio": round(null_ratio, 2),
    }
