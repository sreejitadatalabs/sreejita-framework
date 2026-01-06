# =====================================================
# DATASET SHAPE DETECTOR — UNIVERSAL (STABILIZED)
# Sreejita Framework v3.6
# =====================================================

from enum import Enum
from typing import Dict, Any
import pandas as pd


class DatasetShape(str, Enum):
    """
    Dataset SHAPE describes the structural nature of the data,
    NOT the business domain.
    """
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"


def _norm(col: str) -> str:
    return col.lower().strip().replace(" ", "_").replace("-", "_")


def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect dataset structural shape.

    GUARANTEES:
    - Never raises
    - Conservative (UNKNOWN > wrong)
    - Healthcare-safe
    - Explainable
    """

    try:
        if df is None or df.empty:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": 0.0,
                "signals": {},
                "reason": "Empty dataset",
            }

        cols = {_norm(c) for c in df.columns}

        score = {
            DatasetShape.ROW_LEVEL_CLINICAL: 0.0,
            DatasetShape.AGGREGATED_OPERATIONAL: 0.0,
            DatasetShape.FINANCIAL_SUMMARY: 0.0,
            DatasetShape.QUALITY_METRICS: 0.0,
        }

        reasons = {k: [] for k in score}

        # =================================================
        # HARD ANCHORS (STRONG SIGNALS)
        # =================================================

        if any(
            any(tok in c for tok in ["patient", "mrn", "subject", "person"])
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.6
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append("patient identifier columns")

        if any(
            any(tok in c for tok in ["admission", "discharge", "visit", "encounter"])
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.4
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append("encounter lifecycle dates")

        # =================================================
        # QUALITY METRICS (CLINICAL OUTCOME FOCUSED)
        # =================================================

        if any("rate" in c or "ratio" in c for c in cols):
            score[DatasetShape.QUALITY_METRICS] += 0.4
            reasons[DatasetShape.QUALITY_METRICS].append("rate/ratio metrics")

        if any(
            any(tok in c for tok in ["readmission", "mortality", "infection", "adverse"])
            for c in cols
        ):
            score[DatasetShape.QUALITY_METRICS] += 0.6
            reasons[DatasetShape.QUALITY_METRICS].append("clinical outcome indicators")

        # =================================================
        # AGGREGATED OPERATIONAL (NON-CLINICAL)
        # =================================================

        if any(
            any(tok in c for tok in ["total", "volume", "count", "throughput"])
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.5
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append("aggregate volume metrics")

        if any(
            any(tok in c for tok in ["avg", "average", "mean", "median"])
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.5
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append("summary statistics")

        # =================================================
        # FINANCIAL SUMMARY (STRICT)
        # =================================================

        financial_amount = any(
            any(tok in c for tok in ["revenue", "expense", "profit", "margin"])
            for c in cols
        )

        financial_grouping = any(
            any(tok in c for tok in ["cost_center", "gl_code", "ledger"])
            for c in cols
        )

        # Cost alone is NOT enough (healthcare-safe)
        if financial_amount and financial_grouping:
            score[DatasetShape.FINANCIAL_SUMMARY] += 1.0
            reasons[DatasetShape.FINANCIAL_SUMMARY].append(
                "financial amounts with accounting groupings"
            )

        # =================================================
        # CONFLICT RESOLUTION (CRITICAL)
        # =================================================

        # If patient-level data exists, demote finance & ops
        if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 0.6:
            score[DatasetShape.FINANCIAL_SUMMARY] *= 0.3
            score[DatasetShape.AGGREGATED_OPERATIONAL] *= 0.5

        # =================================================
        # FINAL DECISION
        # =================================================

        best_shape = max(score, key=score.get)
        best_score = score[best_shape]

        if best_score < 0.6:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": round(best_score, 2),
                "signals": score,
                "reason": "No dominant structural pattern",
            }

        return {
            "shape": best_shape,
            "confidence": round(min(best_score, 1.0), 2),
            "signals": score,
            "reason": "; ".join(reasons[best_shape]) or "Heuristic match",
        }

    except Exception:
        return {
            "shape": DatasetShape.UNKNOWN,
            "confidence": 0.0,
            "signals": {},
            "reason": "Shape detection failed safely",
        }
