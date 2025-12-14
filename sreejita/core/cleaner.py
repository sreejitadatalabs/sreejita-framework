import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame, preserve_date_cols: list = None):
    """
    Clean a dataframe by standardizing column names, removing duplicates,
    and cleaning whitespace.
    
    Args:
        df: Input dataframe
        preserve_date_cols: List of columns to preserve as-is (date columns)
    
    Returns:
        dict with 'df' (cleaned dataframe) and 'summary' (cleaning summary)
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(" ", "_")
    )
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Replace empty strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)
    
    # Clean whitespace in object columns
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return {
        "df": df,
        "summary": {
            "rows": len(df),
            "columns": df.shape[1],
            "dtypes": df.dtypes.to_dict()
        }
    }
