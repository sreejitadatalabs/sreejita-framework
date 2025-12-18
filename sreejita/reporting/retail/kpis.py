from sreejita.core.kpi_utils import safe_sum
import pandas as pd

def compute_customer_kpis(df: pd.DataFrame):
    kpis = {}

    revenue = safe_sum(df, "revenue")
    if revenue is not None:
        kpis["total_revenue"] = {
            "value": round(revenue, 2),
            "label": "Total Revenue",
            "unit": "currency"
        }

    return kpis
