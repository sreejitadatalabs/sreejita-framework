from sreejita.core.kpi_utils import safe_mean
import pandas as pd

def compute_ops_kpis(df: pd.DataFrame):
    kpis = {}

    avg_delivery = safe_mean(df, "delivery_time")
    if avg_delivery is not None:
        kpis["avg_delivery_time"] = {
            "value": round(avg_delivery, 2),
            "label": "Average Delivery Time",
            "unit": "days"
        }

    return kpis
