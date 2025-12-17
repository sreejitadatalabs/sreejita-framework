from typing import Dict
import pandas as pd


COMMON_SALES_COLUMNS = [
    "sales",
    "Sales",
    "revenue",
    "Revenue",
    "order_amount",
    "Order Amount",
    "total_sales",
    "Total Sales",
]


def _detect_sales_column(df: pd.DataFrame) -> str:
    for col in COMMON_SALES_COLUMNS:
        if col in df.columns:
            return col
    return None


def compute_retail_kpis(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Retail KPIs (v2.8.3)
    Config-driven with safe auto-detection fallback
    """

    dataset_cfg = config.get("dataset", {})

    sales_col = dataset_cfg.get("sales")

    # --------------------------------------------------
    # SAFE FALLBACK (for UI / quick runs)
    # --------------------------------------------------
    if not sales_col:
        sales_col = _detect_sales_column(df)

    if not sales_col or sales_col not in df.columns:
        raise KeyError(
            "No sales column found. "
            "Please specify dataset.sales in config.yaml "
            "or use a standard column name (Sales, Revenue, etc.)"
        )

    sales = df[sales_col]

    kpis = {
        "total_sales": float(sales.sum()),
        "order_count": int(len(df)),
        "average_order_value": float(sales.mean()),
    }

    # Optional metrics
    shipping_col = dataset_cfg.get("shipping")
    profit_col = dataset_cfg.get("profit")
    discount_col = dataset_cfg.get("discount")

    if shipping_col and shipping_col in df.columns:
        kpis["shipping_cost_ratio"] = float(
            df[shipping_col].sum() / sales.sum()
        )
        kpis["target_shipping_cost_ratio"] = 0.09

    if profit_col and profit_col in df.columns:
        kpis["profit_margin"] = float(
            df[profit_col].sum() / sales.sum()
        )
        kpis["target_profit_margin"] = 0.12

    if discount_col and discount_col in df.columns:
        kpis["average_discount"] = float(df[discount_col].mean())

    return kpis
