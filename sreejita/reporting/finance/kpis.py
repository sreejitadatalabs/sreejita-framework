def compute_finance_kpis(df):
    return {
        "total_revenue": df["revenue"].sum(),
        "net_margin": (df["revenue"].sum() - df["expenses"].sum()) / df["revenue"].sum(),
        "expense_ratio": df["expenses"].sum() / df["revenue"].sum()
    }
