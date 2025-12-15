"""
Finance Domain Module
"""

from typing import Dict, List, Any, Set
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Financial analytics"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return isinstance(df, pd.DataFrame)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}
        if "revenue" in df.columns:
            kpis["Total Revenue"] = df["revenue"].sum()
        if "expenses" in df.columns:
            kpis["Total Expenses"] = df["expenses"].sum()
        return kpis

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        return [f"{k}: {v}" for k, v in kpis.items()]


class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"

    FINANCE_COLUMNS: Set[str] = {
        "revenue", "expenses", "profit",
        "cost", "balance", "margin"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("finance", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.FINANCE_COLUMNS)

        score = min((len(matches) / len(self.FINANCE_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="finance",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )
