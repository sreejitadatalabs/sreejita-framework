import pandas as pd


def compute_retail_kpis(df: pd.DataFrame, config: dict):
    """
    Compute core Retail KPIs.

    Expected config structure:
    config:
      dataset:
        sales: <column name>
        profit: <column name> (optional)
        shipping_cost: <column name> (optional)
    """

    dataset_cfg = config.get("dataset", {})

    sales_col = dataset_cfg.get("sales")
    profit_col = dataset_cfg.get("profit")
    shipping_col = dataset_cfg.get("shipping_cost")

    if not sales_col or sales_col not in df.columns:
        raise KeyError(
            f"Configured sales column '{sales_col}' not found in dataset"
        )

    sales = df[sales_col]

    kpis = {
        "total_sales": float(sales.sum()),
        "order_count": int(len(df)),
        "average_order_value": float(sales.mean()),
    }

    # Optional profit KPIs
    if profit_col and profit_col in df.columns:
        profit = df[profit_col]
        kpis["total_profit"] = float(profit.sum())
        kpis["profit_margin"] = float(profit.sum() / sales.sum()) if sales.sum() else 0.0

    # Optional shipping KPIs
    if shipping_col and shipping_col in df.columns:
        shipping = df[shipping_col]
        kpis["shipping_cost_ratio"] = float(shipping.sum() / sales.sum()) if sales.sum() else 0.0

    return kpis
