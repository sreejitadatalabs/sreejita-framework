from sreejita.core.kpi_utils import safe_sum
import pandas as pd

def compute_marketing_kpis(df: pd.DataFrame):
    kpis = {}

    total_cost = safe_sum(df, "cost")
    if total_cost is not None:
        kpis["total_cost"] = {
            "value": round(total_cost, 2),
            "label": "Total Campaign Cost",
            "unit": "currency"
        }

    return kpis
