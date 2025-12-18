def compute_finance_kpis(df):
    return {
        "total_revenue": df["revenue"].sum(),
        "total_cost": df["cost"].sum(),
        "net_profit": df["revenue"].sum() - df["cost"].sum(),
        "profit_margin": (
            (df["revenue"].sum() - df["cost"].sum()) / df["revenue"].sum()
            if df["revenue"].sum() else 0
        ),
    }
