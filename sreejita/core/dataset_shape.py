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


def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects dataset SHAPE (not domain).

    GUARANTEES:
    - Never raises
    - Deterministic
    - Explainable
    - Safe for messy, partial, real-world data
    """

    try:
        # Normalize columns aggressively for fuzzy matching
        cols = {
            c.lower().strip().replace(" ", "_").replace("-", "_")
            for c in df.columns
        }
        row_count = len(df)

        score = {
            DatasetShape.ROW_LEVEL_CLINICAL: 0,
            DatasetShape.AGGREGATED_OPERATIONAL: 0,
            DatasetShape.FINANCIAL_SUMMARY: 0,
            DatasetShape.QUALITY_METRICS: 0,
        }

        reasons = {k: [] for k in score}

        # =================================================
        # ROW-LEVEL CLINICAL
        # =================================================
        if any(tok in c for c in cols for tok in ["patient", "mrn", "subject", "person"]):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append("patient identifiers present")

        if any(tok in c for c in cols for tok in ["admission", "discharge", "visit_date", "encounter"]):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append("admission/discharge dates present")

        if row_count > 300:
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 1
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append("high row count")

        # =================================================
        # AGGREGATED OPERATIONAL
        # =================================================
        if any(tok in c for c in cols for tok in ["total", "volume", "visits", "count"]):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append("aggregated volume fields present")

        if any(tok in c for c in cols for tok in ["avg", "average", "mean", "median"]):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 2
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append("average/summary metrics present")

        if row_count < 300:
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 1
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append("low-to-moderate row count")

        # =================================================
        # FINANCIAL SUMMARY
        # =================================================
        if any(tok in c for c in cols for tok in ["revenue", "billing", "charge", "cost", "expense"]):
            score[DatasetShape.FINANCIAL_SUMMARY] += 3
            reasons[DatasetShape.FINANCIAL_SUMMARY].append("financial amount fields present")

        if any(tok in c for c in cols for tok in ["department", "service", "cost_center"]):
            score[DatasetShape.FINANCIAL_SUMMARY] += 2
            reasons[DatasetShape.FINANCIAL_SUMMARY].append("financial grouping fields present")

        # =================================================
        # QUALITY METRICS
        # =================================================
        if any("rate" in c or "ratio" in c for c in cols):
            score[DatasetShape.QUALITY_METRICS] += 2
            reasons[DatasetShape.QUALITY_METRICS].append("rate/ratio metrics present")

        if any(tok in c for c in cols for tok in ["readmission", "mortality", "infection", "adverse"]):
            score[DatasetShape.QUALITY_METRICS] += 3
            reasons[DatasetShape.QUALITY_METRICS].append("clinical quality indicators present")

        # =================================================
        # FINAL DECISION
        # =================================================
        best_shape = max(score, key=score.get)
        max_score = score[best_shape]

        if max_score == 0:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": 0.0,
                "signals": score,
                "reason": "No strong structural signals detected",
            }

        # Confidence normalized against reasonable maximum (5)
        confidence = round(min(max_score / 5.0, 1.0), 2)

        return {
            "shape": best_shape,
            "confidence": confidence,
            "signals": score,
            "reason": "; ".join(reasons[best_shape]) or "Heuristic match",
        }

    except Exception:
        # Absolute safety fallback
        return {
            "shape": DatasetShape.UNKNOWN,
            "confidence": 0.0,
            "signals": {},
            "reason": "Shape detection failed safely",
        }
