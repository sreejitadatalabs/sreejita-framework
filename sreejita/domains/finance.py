from typing import Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Finance analytics domain"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return isinstance(df, pd.DataFrame) and {"revenue", "cost"}.issubset(df.columns)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------
    # REQUIRED ABSTRACT METHODS
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame):
        from sreejita.reporting.finance.kpis import compute_finance_kpis
        return compute_finance_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis):
        from sreejita.reporting.finance.insights import generate_finance_insights
        return generate_finance_insights(df, kpis)


class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"

    FINANCE_COLUMNS: Set[str] = {
        "revenue",
        "cost",
        "expense",
        "profit",
        "cash",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("finance", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.FINANCE_COLUMNS)

        confidence = min(len(matches) / len(self.FINANCE_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="finance",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


def register(registry):
    registry.register(
        name="finance",
        domain_cls=FinanceDomain,
        detector_cls=FinanceDomainDetector,
    )
