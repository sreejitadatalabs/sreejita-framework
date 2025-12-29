from enum import Enum
from typing import Dict, Any
import pandas as pd


class DatasetShape(str, Enum):
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"


def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects dataset SHAPE (not domain).

    This function MUST:
    - Never raise
    - Be explainable
    - Work with partial / messy data
    """

    cols = set(c.lower() for c in df.columns)
    row_count = len(df)

    score = {
        DatasetShape.ROW_LEVEL_CLINICAL: 0,
        DatasetShape.AGGREGATED_OPERATIONAL: 0,
        DatasetShape.FINANCIAL_SUMMARY: 0,
        DatasetShape.QUALITY_METRICS: 0,
    }

    # ---------- ROW-LEVEL CLINICAL ----------
    if any(c in cols for c in ["patient_id", "mrn"]):
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    if any("admission" in c or "discharge" in c for c in cols):
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    if row_count > 500:
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 1

    # ---------- AGGREGATED OPERATIONAL ----------
    if any(c in cols for c in ["total_patients", "visits", "volume"]):
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
    if any(c in cols for c in ["avg_los", "average_los", "bed_occupancy"]):
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 2
    if row_count < 500:
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 1

    # ---------- FINANCIAL SUMMARY ----------
    if any(c in cols for c in ["revenue", "billing", "charges", "total_cost"]):
        score[DatasetShape.FINANCIAL_SUMMARY] += 2
    if any(c in cols for c in ["service", "department", "cost_center"]):
        score[DatasetShape.FINANCIAL_SUMMARY] += 2

    # ---------- QUALITY METRICS ----------
    if any("rate" in c for c in cols):
        score[DatasetShape.QUALITY_METRICS] += 1
    if any(c in cols for c in ["readmission_rate", "mortality_rate"]):
        score[DatasetShape.QUALITY_METRICS] += 3

    # ---------- FINAL DECISION ----------
    best_shape = max(score, key=score.get)
    confidence = score[best_shape] / max(sum(score.values()), 1)

    if score[best_shape] == 0:
        best_shape = DatasetShape.UNKNOWN
        confidence = 0.0

    return {
        "shape": best_shape,
        "confidence": round(confidence, 2),
        "signals": score,
    }
