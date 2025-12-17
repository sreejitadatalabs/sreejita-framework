from typing import Dict
import pandas as pd


def compute_retail_kpis(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Retail KPIs (v2.8+)
    Config-driven, deterministic, executive-safe
    """

    dataset_cfg = config.get("dataset", {})

    sales_col = dataset_cfg.get("sales")
    shipping_col = dataset_cfg.get("shipping")
    discount_col = dataset_cfg.get("discount")
    profit_col = dataset_cfg.get("profit")

    if not sales_col or sales_col not in df.columns:
        raise KeyError(
            f"Configured sales column '{sales_col}' not found in dataset"
        )

    sales = df[sales_col]

    kpis = {
        # Scale
        "total_sales": float(sales.sum()),
        "order_count": int(len(df)),

        # Efficiency
        "average_order_value": float(sales.mean()),
    }

    # -------------------------
    # Shipping Cost Ratio
    # -------------------------
    if shipping_col and shipping_col in df.columns:
        shipping = df[shipping_col]
        shipping_ratio = shipping.sum() / sales.sum()

        kpis["shipping_cost_ratio"] = float(shipping_ratio)
        kpis["target_shipping_cost_ratio"] = 0.09

    # -------------------------
    # Profit Margin
    # -------------------------
    if profit_col and profit_col in df.columns:
        profit = df[profit_col]
        profit_margin = profit.sum() / sales.sum()

        kpis["profit_margin"] = float(profit_margin)
        kpis["target_profit_margin"] = 0.12

    # -------------------------
    # Discount Behavior
    # -------------------------
    if discount_col and discount_col in df.columns:
        discount = df[discount_col]
        kpis["average_discount"] = float(discount.mean())

    return kpis
