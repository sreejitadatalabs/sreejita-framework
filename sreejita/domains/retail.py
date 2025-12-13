def enrich(df):
    if "profit" in df.columns and "sales" in df.columns:
        df["margin"] = df["profit"] / df["sales"]
    return df
