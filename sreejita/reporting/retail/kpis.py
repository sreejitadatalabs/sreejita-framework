def compute_retail_kpis(df):
    """
    Domain-aware Retail KPIs (v2.8)
    Deterministic, target-aware, executive-ready
    """

    # -------------------------
    # Core columns
    # -------------------------
    sales = df["sales"]
    shipping = df.get("shipping_cost")
    discount = df.get("discount")
    profit = df.get("profit")

    # -------------------------
    # Base KPIs
    # -------------------------
    kpis = {
        # Scale metrics
        "total_sales": float(sales.sum()),
        "order_count": int(len(df)),

        # Efficiency metrics
        "average_order_value": float(sales.mean()),
    }

    # -------------------------
    # Shipping Cost Ratio
    # -------------------------
    if shipping is not None:
        shipping_ratio = shipping.sum() / sales.sum()
        kpis["shipping_cost_ratio"] = float(shipping_ratio)

        # Target (context)
        kpis["target_shipping_cost_ratio"] = 0.09

    # -------------------------
    # Profit Margin
    # -------------------------
    if profit is not None:
        profit_margin = profit.sum() / sales.sum()
        kpis["profit_margin"] = float(profit_margin)

        # Target (context)
        kpis["target_profit_margin"] = 0.12

    # -------------------------
    # Discount Behavior
    # -------------------------
    if discount is not None:
        kpis["average_discount"] = float(discount.mean())

    return kpis
