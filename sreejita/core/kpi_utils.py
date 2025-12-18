import pandas as pd

def has_columns(df: pd.DataFrame, *cols) -> bool:
    return all(col in df.columns for col in cols)

def safe_sum(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    if df[col].dropna().empty:
        return None
    return float(df[col].sum())

def safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    if df[col].dropna().empty:
        return None
    return float(df[col].mean())
