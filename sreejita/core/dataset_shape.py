# =====================================================
# DATASET SHAPE DETECTOR — UNIVERSAL (LOCKED)
# Sreejita Framework v3.6 STABILIZED
# =====================================================

from enum import Enum
from typing import Dict, Any
import pandas as pd


# =====================================================
# DATASET SHAPE ENUM (STRUCTURE ONLY — NOT DOMAIN)
# =====================================================

class DatasetShape(str, Enum):
    """
    Dataset SHAPE describes STRUCTURE, never business domain.
    """
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"


# =====================================================
# NORMALIZATION (STRICT, NON-FUZZY)
# =====================================================

def _norm(col: str) -> str:
    col = str(col).lower().strip()
    col = col.replace(" ", "_").replace("-", "_")
    return col


# =====================================================
# SHAPE DETECTION (CONSERVATIVE & EXPLAINABLE)
# =====================================================

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect dataset STRUCTURAL shape.

    GUARANTEES:
    - Never raises
    - Conservative (UNKNOWN > wrong)
    - Healthcare-safe
    - Zero sub-domain inference
    """

    try:
        # -----------------------------
        # BASIC SAFETY
        # -----------------------------
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": 0.0,
                "signals": {},
                "reason": "Empty or invalid dataset",
            }

        cols = {_norm(c) for c in df.columns}
        row_count = len(df)

        # -----------------------------
        # SCORE BUCKETS (0–1 RANGE)
        # -----------------------------
        score = {
            DatasetShape.ROW_LEVEL_CLINICAL: 0.0,
            DatasetShape.AGGREGATED_OPERATIONAL: 0.0,
            DatasetShape.FINANCIAL_SUMMARY: 0.0,
            DatasetShape.QUALITY_METRICS: 0.0,
        }

        reasons = {k: [] for k in score}

        # =================================================
        # ROW-LEVEL CLINICAL (HARD ANCHORS)
        # =================================================

        if any(
            any(tok in c for tok in ("patient", "mrn", "subject", "person"))
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.55
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                "patient-level identifiers present"
            )

        if any(
            any(tok in c for tok in ("admission", "discharge", "visit", "encounter"))
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.35
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                "encounter lifecycle timestamps present"
            )

        if row_count > 100:
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.10
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                "sufficient row volume for row-level analysis"
            )

        # =================================================
        # QUALITY METRICS (OUTCOME-FOCUSED)
        # =================================================

        if any("rate" in c or "ratio" in c for c in cols):
            score[DatasetShape.QUALITY_METRICS] += 0.4
            reasons[DatasetShape.QUALITY_METRICS].append(
                "rate or ratio metrics detected"
            )

        if any(
            any(tok in c for tok in ("readmission", "mortality", "infection", "adverse"))
            for c in cols
        ):
            score[DatasetShape.QUALITY_METRICS] += 0.6
            reasons[DatasetShape.QUALITY_METRICS].append(
                "clinical outcome indicators present"
            )

        # =================================================
        # AGGREGATED OPERATIONAL (NON-CLINICAL)
        # =================================================

        if any(
            any(tok in c for tok in ("total", "volume", "count", "throughput"))
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.5
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append(
                "aggregate volume metrics detected"
            )

        if any(
            any(tok in c for tok in ("avg", "average", "mean", "median"))
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.5
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append(
                "summary statistics detected"
            )

        # =================================================
        # FINANCIAL SUMMARY (STRICT — NO COST LEAKAGE)
        # =================================================

        financial_amount = any(
            any(tok in c for tok in ("revenue", "expense", "profit", "margin"))
            for c in cols
        )

        financial_grouping = any(
            any(tok in c for tok in ("cost_center", "gl_code", "ledger"))
            for c in cols
        )

        # 🔒 COST ALONE IS NOT A FINANCIAL SUMMARY
        if financial_amount and financial_grouping:
            score[DatasetShape.FINANCIAL_SUMMARY] += 1.0
            reasons[DatasetShape.FINANCIAL_SUMMARY].append(
                "financial amounts with accounting groupings"
            )

        # =================================================
        # CONFLICT RESOLUTION (CRITICAL FIX)
        # =================================================

        # Patient-level data dominates ALL other interpretations
        if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 0.6:
            score[DatasetShape.FINANCIAL_SUMMARY] *= 0.25
            score[DatasetShape.AGGREGATED_OPERATIONAL] *= 0.5

        # Quality metrics override aggregated ops
        if score[DatasetShape.QUALITY_METRICS] >= 0.6:
            score[DatasetShape.AGGREGATED_OPERATIONAL] *= 0.5

        # =================================================
        # FINAL DECISION (CONSERVATIVE)
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
        # 🔒 ABSOLUTE SAFETY FALLBACK
        return {
            "shape": DatasetShape.UNKNOWN,
            "confidence": 0.0,
            "signals": {},
            "reason": "Shape detection failed safely",
        }
