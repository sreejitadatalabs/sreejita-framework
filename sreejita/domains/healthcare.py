from typing import Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Healthcare analytics domain"

    def validate_data(self, df: pd.DataFrame) -> bool:
        required = {"outcome_score", "readmitted"}
        return isinstance(df, pd.DataFrame) and required.issubset(df.columns)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------
    # REQUIRED ABSTRACT METHODS
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame):
        from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis
        return compute_healthcare_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis):
        from sreejita.reporting.healthcare.insights import generate_healthcare_insights
        return generate_healthcare_insights(df, kpis)


class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient_id",
        "outcome_score",
        "readmitted",
        "length_of_stay",
        "diagnosis",
        "treatment",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("healthcare", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.HEALTHCARE_COLUMNS)

        confidence = min(len(matches) / len(self.HEALTHCARE_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
