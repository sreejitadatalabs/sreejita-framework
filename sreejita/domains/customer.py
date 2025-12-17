from typing import Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Customer analytics domain"

    # -------------------------
    # Validation / Preprocess
    # -------------------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return isinstance(df, pd.DataFrame) and "customer_id" in df.columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------
    # REQUIRED ABSTRACT METHODS
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame):
        # 🔥 Lazy import — SAFE
        from sreejita.reporting.customer.kpis import compute_customer_kpis
        return compute_customer_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis):
        # 🔥 Lazy import — SAFE
        from sreejita.reporting.customer.insights import generate_customer_insights
        return generate_customer_insights(df, kpis)


class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "customer"

    CUSTOMER_COLUMNS: Set[str] = {
        "customer_id",
        "customer_name",
        "email",
        "phone",
        "segment",
        "revenue",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("customer", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.CUSTOMER_COLUMNS)

        confidence = min(len(matches) / len(self.CUSTOMER_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="customer",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


def register(registry):
    registry.register(
        name="customer",
        domain_cls=CustomerDomain,
        detector_cls=CustomerDomainDetector,
    )
