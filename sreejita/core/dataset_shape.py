from enum import Enum
from typing import Dict, Any
import pandas as pd


# =====================================================
# DATASET SHAPE ENUM
# =====================================================

class DatasetShape(str, Enum):
    """
    Structural shape of the dataset.
    CONTEXT only — never business logic.
    """
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    UNKNOWN = "unknown"


# =====================================================
# DATASET SHAPE DETECTOR (DETERMINISTIC)
# =====================================================

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects dataset structure using column signals only.

    Rules:
    - No domain assumptions
    - No thresholds
    - No ML
    - Deterministic & explainable
    """

    if df is None or df.empty:
        return {
            "shape": DatasetShape.UNKNOWN,
            "score": {},
            "signals": {}
        }

    cols = [
        c.lower().strip().replace(" ", "_")
        for c in df.columns
    ]

    score = {
        DatasetShape.ROW_LEVEL_CLINICAL: 0,
        DatasetShape.AGGREGATED_OPERATIONAL: 0,
        DatasetShape.FINANCIAL_SUMMARY: 0,
        DatasetShape.UNKNOWN: 0,
    }

    signals = {
        "patient_identifier": False,
        "time_dimension": False,
        "aggregation_keywords": False,
        "financial_keywords": False,
        "entity_keywords": False,     # NEW
        "population_keywords": False, # NEW
    }

    # -------------------------------------------------
    # ROW-LEVEL CLINICAL SIGNALS
    # -------------------------------------------------
    if any(
        any(k in c for k in [
            "patient", "mrn", "pid",
            "encounter", "visit", "admission"
        ])
        for c in cols
    ):
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
        signals["patient_identifier"] = True

    if any(
        any(k in c for k in [
            "date", "time", "admit",
            "arrival", "discharge", "sample_date"
        ])
        for c in cols
    ):
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
        signals["time_dimension"] = True

    # -------------------------------------------------
    # ENTITY-LEVEL (DIAGNOSTICS / FACILITY / REGION)
    # -------------------------------------------------
    if any(
        any(k in c for k in [
            "test", "lab", "specimen",
            "facility", "center", "provider"
        ])
        for c in cols
    ):
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 1
        signals["entity_keywords"] = True

    # -------------------------------------------------
    # POPULATION / PUBLIC HEALTH SIGNALS
    # -------------------------------------------------
    if any(
        any(k in c for k in [
            "population", "region", "district",
            "state", "country", "incidence", "prevalence"
        ])
        for c in cols
    ):
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 2
        signals["population_keywords"] = True

    # -------------------------------------------------
    # AGGREGATED OPERATIONAL SIGNALS
    # -------------------------------------------------
    if any(
        any(k in c for k in [
            "total", "count", "volume",
            "census", "avg", "mean", "rate"
        ])
        for c in cols
    ):
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
        signals["aggregation_keywords"] = True

    # -------------------------------------------------
    # FINANCIAL SUMMARY SIGNALS
    # -------------------------------------------------
    if any(
        any(k in c for k in [
            "revenue", "cost", "billing",
            "charges", "amount", "expense",
            "claim", "payment"
        ])
        for c in cols
    ):
        score[DatasetShape.FINANCIAL_SUMMARY] += 3
        signals["financial_keywords"] = True

    # -------------------------------------------------
    # FINAL DECISION (PRIORITIZED)
    # -------------------------------------------------
    if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 4:
        shape = DatasetShape.ROW_LEVEL_CLINICAL
    else:
        shape = max(score, key=score.get)

    if score[shape] == 0:
        shape = DatasetShape.UNKNOWN

    return {
        "shape": shape,
        "score": score,
        "signals": signals
    }
