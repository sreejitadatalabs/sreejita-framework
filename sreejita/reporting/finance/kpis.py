def compute_finance_kpis(df):
    """
    Finance KPIs (DEFENSIVE)

    If required columns are missing, return empty KPIs.
    """
    required_cols = {"revenue", "expenses"}

    if not required_cols.issubset(df.columns):
        return {}

    total_revenue = df["revenue"].sum()
    total_expenses = df["expenses"].sum()

    return {
        "total_revenue": float(total_revenue),
        "net_margin": float(
            (total_revenue - total_expenses) / total_revenue
        ) if total_revenue else 0.0,
        "expense_ratio": float(
            total_expenses / total_revenue
        ) if total_revenue else 0.0,
    }
