"""
Customer Domain
---------------
Domain brain for Customer Analytics.
Delegates all computation to reporting.customer.*
"""

from typing import Dict, List, Any, Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

from sreejita.reporting.customer.kpis import compute_customer_kpis
from sreejita.reporting.customer.insights import generate_insights
from sreejita.reporting.customer.recommendations import generate_recommendations
from sreejita.reporting.customer.visuals import generate_visuals

from sreejita.core.decision_snapshot import DecisionSnapshot


# ---------------------------------------------------------------------
# Domain Implementation
# ---------------------------------------------------------------------

class CustomerDomain(BaseDomain):
    """
    Customer Analytics Domain
    """

    name = "customer"
    description = "Customer behavior, retention, churn, and value analytics"

    # -------------------------
    # Validation & Preprocess
    # -------------------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and "customer_id" in df.columns
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Minimal, safe preprocessing
        return df.copy()

    # -------------------------
    # Domain Pipeline
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Delegate KPI computation to reporting layer.
        """
        return compute_customer_kpis(df)

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Dict[str, Any]],
        snapshot: DecisionSnapshot | None = None,
    ) -> List[Any]:
        """
        Generate customer insights (rule-based).
        """
        return generate_insights(kpis=kpis, snapshot=snapshot)

    def generate_recommendations(
        self,
        insights: List[Any],
    ) -> List[Any]:
        """
        Generate customer recommendations.
        """
        return generate_recommendations(insights)

    def generate_visuals(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Dict[str, Any]],
    ):
        """
        Generate customer visuals.
        """
        return generate_visuals(df=df, kpis=kpis)


# ---------------------------------------------------------------------
# Domain Detector
# ---------------------------------------------------------------------

class CustomerDomainDetector(BaseDomainDetector):
    """
    Detects Customer domain based on column signals.
    """

    domain_name = "customer"

    CUSTOMER_COLUMNS: Set[str] = {
        "customer_id",
        "customer_name",
        "email",
        "phone",
        "segment",
        "revenue",
        "transaction_date",
        "orders",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult(
                domain="customer",
                confidence=0.0,
                signals={"reason": "invalid_dataframe"},
            )

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.CUSTOMER_COLUMNS)

        confidence = min(len(matches) / len(self.CUSTOMER_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="customer",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


# ---------------------------------------------------------------------
# Registration Hook
# ---------------------------------------------------------------------

def register(registry):
    """
    Auto-registration hook for domain registry.
    """
    registry.register(
        name="customer",
        domain_cls=CustomerDomain,
        detector_cls=CustomerDomainDetector,
    )
