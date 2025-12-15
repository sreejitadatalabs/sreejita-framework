"""
Schema detection module for automatic dataframe role inference.

Purpose:
- Distinguish true numeric MEASURES from identifiers/codes
- Prevent meaningless analysis (postal_code, row_id, etc.)
- Serve as foundation for v2.0 Domain Intelligence
"""

import pandas as pd
from typing import Dict, List

ID_LIKE_KEYWORDS = {
    "id", "code", "postal", "zip", "pin", "sku", "number", "no"
}


def _is_id_like(column_name: str) -> bool:
    name = column_name.lower()
    return any(k in name for k in ID_LIKE_KEYWORDS)


def detect_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_measures = []
    categorical = []
    datetime_cols = []
    identifiers = []

    for col in df.columns:
        series = df[col]

        # ---- Identifier (highest priority) ----
        if _is_id_like(col):
            identifiers.append(col)
            continue

        # ---- Datetime ----
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # ---- Numeric ----
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            if unique_ratio > 0.9:
                identifiers.append(col)
            else:
                numeric_measures.append(col)
            continue

        # ---- Categorical ----
        categorical.append(col)

    return {
        "numeric_measures": numeric_measures,
        "categorical": categorical,
        "datetime": datetime_cols,
        "identifiers": identifiers,
    }


__all__ = ["detect_schema"]
