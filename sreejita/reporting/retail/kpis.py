import pandas as pd
from typing import Dict, Any

def compute_retail_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Domain-aware Retail KPIs (v2.x)
    Defensive, deterministic, report-safe
    """

    kpis = {}

    # -------------------------
    # Required base metric: sales
    # -------------------------
    if "sales" not in df.columns or df["sales"].dropna().empty:
        return kpis  # No sales → no retail KPIs (but NO crash)

    sales = df["sales"]
    sales_sum = sales.sum()

    if sales_sum == 0:
        return kpis

    # -------------------------
    # Base KPIs
    # -------------------------
    kpis["total_sales"] = float(sales_sum)
    kpis["order_count"] = int(len(df))
    kpis["average_order_value"] = float(sales.mean())

    # -------------------------
    # Shipping Cost Ratio
    # -------------------------
    if "shipping_cost" in df.columns:
        shipping = df["shipping_cost"].dropna()
        if not shipping.empty:
            kpis["shipping_cost_ratio"] = float(shipping.sum() / sales_sum)
            kpis["target_shipping_cost_ratio"] = 0.09

    # -------------------------
    # Profit Margin
    # -------------------------
    if "profit" in df.columns:
        profit = df["profit"].dropna()
        if not profit.empty:
            kpis["profit_margin"] = float(profit.sum() / sales_sum)
            kpis["target_profit_margin"] = 0.12

    # -------------------------
    # Discount Behavior
    # -------------------------
    if "discount" in df.columns:
        discount = df["discount"].dropna()
        if not discount.empty:
            kpis["average_discount"] = float(discount.mean())

    return kpis
