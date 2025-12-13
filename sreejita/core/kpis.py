def compute_kpis(df, sales_col="sales", profit_col="profit"):
    kpis = {}

    if sales_col in df.columns:
        kpis["Total Sales"] = f"${df[sales_col].sum():,.2f}"
        kpis["Avg Order Value"] = f"${df[sales_col].mean():,.2f}"

    if profit_col in df.columns and sales_col in df.columns:
        kpis["Total Profit"] = f"${df[profit_col].sum():,.2f}"
        kpis["Profit Margin"] = f"{(df[profit_col].sum() / df[sales_col].sum())*100:.1f}%"

    kpis["Rows"] = f"{df.shape[0]:,}"
    kpis["Columns"] = f"{df.shape[1]}"

    return kpis
