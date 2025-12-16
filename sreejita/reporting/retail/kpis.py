def compute_retail_kpis(df):
    """
    Domain-aware Retail KPIs
    """
    sales = df["sales"]
    shipping = df.get("shipping_cost")
    discount = df.get("discount")

    kpis = {
        "total_sales": float(sales.sum()),
        "average_order_value": float(sales.mean()),
        "order_count": int(len(df)),
    }

    if shipping is not None:
        kpis["shipping_cost_ratio"] = float(
            shipping.sum() / sales.sum()
        )

    if discount is not None:
        kpis["average_discount"] = float(discount.mean())

    return kpis
