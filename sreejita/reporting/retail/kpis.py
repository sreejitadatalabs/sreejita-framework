from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio


def compute_retail_kpis(df):
    total_sales = safe_sum(df, "sales") or safe_sum(df, "total_spend")
    total_profit = safe_sum(df, "profit")

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "profit_margin": safe_ratio(total_profit, total_sales),
        "avg_discount": safe_mean(df, "discount"),
        "avg_order_value": safe_mean(df, "final_amount"),
        "total_quantity": safe_sum(df, "quantity"),
    }
