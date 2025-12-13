import pandas as pd
import numpy as np
from typing import Optional, List, Dict

def clean_dataframe(df: pd.DataFrame, date_cols: Optional[List[str]] = None) -> Dict:
    summary = {"rows_before": df.shape[0]}

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[-/]", "_", regex=True)
    )

    df = df.drop_duplicates()

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    df = df.replace(r"^\s*$", np.nan, regex=True)

    # numeric coercion
    for col in df.select_dtypes(include="object"):
        sample = df[col].dropna().astype(str).head(50)
        if not sample.empty and sample.str.replace(",", "").str.match(r"^-?\d+(\.\d+)?$").all():
            df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    summary["rows_after"] = df.shape[0]
    summary["n_nulls_by_col"] = df.isnull().sum().to_dict()
    summary["sample_head"] = df.head(5).to_dict(orient="records")

    return {"df": df.reset_index(drop=True), "summary": summary}
