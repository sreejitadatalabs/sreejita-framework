import pandas as pd


def _auto_detect_sales_column(df: pd.DataFrame) -> str | None:
    """
    Heuristic-based sales column detection (deterministic).
    """
    candidates = [
        "sales",
        "revenue",
        "order_value",
        "total_sales",
        "amount",
    ]

    lower_cols = {c.lower(): c for c in df.columns}

    for key in candidates:
        if key in lower_cols:
            return lower_cols[key]

    return None


def compute_retail_kpis(df: pd.DataFrame, config: dict):
    """
    Compute core Retail KPIs.

    Priority:
    1. Config-defined columns
    2. Deterministic auto-detection
    """

    dataset_cfg = config.get("dataset", {}) if config else {}

    sales_col = dataset_cfg.get("sales")
    profit_col = dataset_cfg.get("profit")
    shipping_col = dataset_cfg.get("shipping_cost")

    # -------------------------
    # SALES COLUMN (MANDATORY)
    # -------------------------
    if not sales_col or sales_col not in df.columns:
        sales_col = _auto_detect_sales_column(df)

    if not sales_col:
        raise KeyError(
            "Unable to detect sales column. "
            "Please specify dataset.sales in config."
        )

    sales = df[sales_col]

    kpis = {
        "total_sales": float(sales.sum()),
        "order_count": int(len(df)),
        "average_order_value": float(sales.mean()),
    }

    # -------------------------
    # OPTIONAL: PROFIT
    # -------------------------
    if profit_col and profit_col in df.columns:
        profit = df[profit_col]
        kpis["total_profit"] = float(profit.sum())
        kpis["profit_margin"] = (
            float(profit.sum() / sales.sum()) if sales.sum() else 0.0
        )

    # -------------------------
    # OPTIONAL: SHIPPING
    # -------------------------
    if shipping_col and shipping_col in df.columns:
        shipping = df[shipping_col]
        kpis["shipping_cost_ratio"] = (
            float(shipping.sum() / sales.sum()) if sales.sum() else 0.0
        )

    return kpis
