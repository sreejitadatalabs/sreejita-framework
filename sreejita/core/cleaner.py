import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame):
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(" ", "_")
    )

    df = df.drop_duplicates()
    df = df.replace(r"^\s*$", np.nan, regex=True)

    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)
