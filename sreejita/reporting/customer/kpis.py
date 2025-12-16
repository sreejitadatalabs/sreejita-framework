def compute_customer_kpis(df):
    return {
        "total_customers": df["customer_id"].nunique(),
        "avg_orders_per_customer": df.groupby("customer_id").size().mean(),
        "churn_proxy_rate": df["last_purchase_days"].gt(180).mean()
    }
