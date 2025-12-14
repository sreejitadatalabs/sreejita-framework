"""Schema detection module for automatic dataframe type inference."""

import pandas as pd


def detect_schema(df: pd.DataFrame):
    """Detect column data types in a dataframe.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        dict: Schema with 'numeric', 'categorical', and 'datetime' column lists
    """
    return {
        "numeric": df.select_dtypes("number").columns.tolist(),
        "categorical": df.select_dtypes("object").columns.tolist(),
        "datetime": df.select_dtypes("datetime").columns.tolist()
    }


__all__ = ["detect_schema"]
