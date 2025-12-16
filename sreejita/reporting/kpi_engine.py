def compute_retail_kpis(df):
    return {
        "total_sales": df["sales"].sum(),
        "average_order_value": df["sales"].mean(),
        "shipping_cost_ratio": (
            df["shipping_cost"].sum() / df["sales"].sum()
        ),
        "discount_rate": df["discount"].mean(),
    }
