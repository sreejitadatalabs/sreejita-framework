import pandas as pd

def detect_schema(df: pd.DataFrame):
    return {
        "numeric": df.select_dtypes("number").columns.tolist(),
        "categorical": df.select_dtypes("object").columns.tolist(),
        "datetime": df.select_dtypes("datetime").columns.tolist()
    }
schema = detect_schema(df)

numeric_cols = config["analysis"].get("numeric") or schema["numeric"]
categorical_cols = config["analysis"].get("categorical") or schema["categorical"]
