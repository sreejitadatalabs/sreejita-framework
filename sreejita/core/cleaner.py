import pandas as pd
import numpy as np


def clean_dataframe(df: pd.DataFrame, preserve_date_cols: list = None):
    """
    Clean a dataframe and produce a data integrity summary.

    This function performs light, deterministic cleaning and reports
    data quality metrics required for audit and review readiness.

    Args:
        df: Input dataframe
        preserve_date_cols: List of columns to preserve as-is (e.g., date columns)

    Returns:
        dict with:
            - 'df': cleaned dataframe
            - 'summary': data quality and structural summary
    """
    preserve_date_cols = preserve_date_cols or []
    df_original = df.copy()
    df = df.copy()

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # -----------------------------
    # Data quality metrics (pre-clean)
    # -----------------------------
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    null_ratio = (
        df.isna().mean().to_dict()
        if total_rows > 0 else {}
    )

    # -----------------------------
    # Drop duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Replace empty strings with NaN
    # -----------------------------
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # -----------------------------
    # Clean whitespace in object columns
    # -----------------------------
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip()

    # -----------------------------
    # Simple outlier signal (numeric only)
    # -----------------------------
    outlier_flags = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty or series.std() == 0:
            outlier_flags[col] = 0
            continue

        z_scores = (series - series.mean()) / series.std()
        outlier_flags[col] = int((z_scores.abs() > 3).sum())

    # -----------------------------
    # Reset index
    # -----------------------------
    df = df.reset_index(drop=True)

    # -----------------------------
    # Summary (audit-friendly)
    # -----------------------------
    summary = {
        "rows_original": total_rows,
        "rows_after_cleaning": len(df),
        "columns": df.shape[1],
        "duplicate_rows_removed": int(duplicate_rows),
        "null_ratio_by_column": null_ratio,
        "outlier_counts_by_column": outlier_flags,
        "dtypes": df.dtypes.to_dict()
    }

    return {
        "df": df,
        "summary": summary
    }
