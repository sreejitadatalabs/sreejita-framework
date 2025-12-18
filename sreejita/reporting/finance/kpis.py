from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio


def compute_finance_kpis(df):
    revenue = safe_sum(df, "revenue") or safe_sum(df, "sales")
    cost = safe_sum(df, "cost") or safe_sum(df, "expense")
    profit = safe_sum(df, "profit")

    return {
        "total_revenue": revenue,
        "total_cost": cost,
        "net_profit": profit,
        "profit_margin": safe_ratio(profit, revenue),
        "avg_transaction_value": safe_mean(df, "amount"),
    }
