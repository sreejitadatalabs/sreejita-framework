from sreejita.core.kpi_utils import safe_mean
import pandas as pd

def compute_healthcare_kpis(df: pd.DataFrame):
    kpis = {}

    avg_outcome = safe_mean(df, "outcome_score")
    if avg_outcome is not None:
        kpis["avg_outcome_score"] = {
            "value": round(avg_outcome, 2),
            "label": "Average Outcome Score",
            "unit": "score"
        }

    return kpis
