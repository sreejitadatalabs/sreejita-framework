def enrich(df):
    if "profit" in df.columns and "sales" in df.columns:
        df["margin"] = df["profit"] / df["sales"]
    return df

def domain_kpis(df):
    out = {}
    if "sales" in df.columns:
        out["Total Sales"] = round(df["sales"].sum(), 2)
    if "profit" in df.columns:
        out["Total Profit"] = round(df["profit"].sum(), 2)
    return out
