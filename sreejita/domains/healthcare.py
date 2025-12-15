"""
Healthcare Domain Module
"""

from typing import Dict, Any, List, Set
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# ---------------- Healthcare Analytics ----------------

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Healthcare analytics"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return isinstance(df, pd.DataFrame)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}
        if "cost" in df.columns:
            kpis["Total Cost"] = df["cost"].sum()
        if "revenue" in df.columns:
            kpis["Total Revenue"] = df["revenue"].sum()
        return kpis

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        return [f"{k}: {v}" for k, v in kpis.items()]


# ---------------- Healthcare Detector ----------------

class HealthcareDomainDetector(BaseDomainDetector):
    """
    Public v1.x domain detector — DO NOT REMOVE
    """

    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient_id",
        "diagnosis",
        "treatment",
        "cost",
        "revenue",
        "doctor",
        "hospital",
        "admission_date",
        "discharge_date",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult(
                domain="healthcare",
                confidence=0.0,
                signals={"reason": "invalid_df"},
            )

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.HEALTHCARE_COLUMNS)

        score = min((len(matches) / len(self.HEALTHCARE_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=score,
            signals={"matched_columns": list(matches)},
        )
