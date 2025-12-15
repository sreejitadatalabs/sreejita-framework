"""
Customer Domain Module
"""

from typing import Dict, List, Any, Set
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Customer analytics"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return isinstance(df, pd.DataFrame) and len(df) > 0

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"Total Customers": len(df)}

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        return [f"Total Customers: {kpis['Total Customers']}"]


class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "customer"

    CUSTOMER_COLUMNS: Set[str] = {
        "customer_id", "customer_name", "segment",
        "email", "phone", "orders", "revenue"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("customer", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.CUSTOMER_COLUMNS)

        score = min((len(matches) / len(self.CUSTOMER_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="customer",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )
