def compute_finance_kpis(df):
    revenue = df["revenue"].sum()
    cost = df["cost"].sum()

    return {
        "total_revenue": revenue,
        "total_cost": cost,
        "net_profit": revenue - cost,
        "profit_margin": (revenue - cost) / revenue if revenue else 0,
    }
