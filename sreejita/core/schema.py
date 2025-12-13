import pandas as pd

def detect_schema(df: pd.DataFrame):
    return {
        "numeric": df.select_dtypes("number").columns.tolist(),
        "categorical": df.select_dtypes("object").columns.tolist(),
        "datetime": df.select_dtypes("datetime").columns.tolist()
    }
