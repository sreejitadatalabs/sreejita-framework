def compute_customer_kpis(df):
    """
    Customer KPIs — Retail-parity contract
    """

    customers = df["customer_id"]
    revenue = df["revenue"]

    kpis = {
        "total_customers": int(customers.nunique()),
        "total_revenue": float(revenue.sum()),
        "average_customer_value": float(revenue.mean()),
    }

    return kpis
