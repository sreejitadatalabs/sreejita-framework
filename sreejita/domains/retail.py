"""
Retail Domain Module - Sales, inventory, and product analytics.
v1.x SAFE: contains both RetailDomain and RetailDomainDetector
"""

from typing import Dict, Any, List, Set
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import (
    BaseDomainDetector,
    DomainDetectionResult,
)

# ---------------------------------------------------------------------
# Existing Retail Domain (DO NOT REMOVE IN v1.x)
# ---------------------------------------------------------------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with retail-specific features."""
    df = df.copy()
    if "profit" in df.columns and "sales" in df.columns:
        df["margin"] = df["profit"] / df["sales"]
    return df


def domain_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate retail-specific KPIs."""
    out: Dict[str, Any] = {}
    if "sales" in df.columns:
        out["Total Sales"] = df["sales"].sum()
    if "profit" in df.columns:
        out["Total Profit"] = df["profit"].sum()
    return out


class RetailDomain(BaseDomain):
    """Retail domain implementation for sales and product analytics."""

    name = "retail"
    description = "Retail analytics: sales, inventory, product performance"
    required_columns = ["sales"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "sales" in df.columns or "revenue" in df.columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return enrich(df)

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        return domain_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        insights: List[str] = []
        if "Total Sales" in kpis:
            insights.append(f"Total Sales: ${kpis['Total Sales']:,.0f}")
        if "Total Profit" in kpis:
            insights.append(f"Total Profit: ${kpis['Total Profit']:,.0f}")
        return insights


# ---------------------------------------------------------------------
# NEW: Retail Domain Detector (Added, not replacing)
# ---------------------------------------------------------------------

class RetailDomainDetector(BaseDomainDetector):
    """
    Retail domain detector (routing / classification).
    Public API for v1.x — DO NOT REMOVE.
    """

    domain_name: str = "retail"

    RETAIL_COLUMNS: Set[str] = {
        "sales",
        "revenue",
        "order_id",
        "product",
        "product_name",
        "sku",
        "quantity",
        "discount",
        "store",
        "category",
        "sub_category",
        "price",
        "profit",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult(
                domain=self.domain_name,
                confidence=0.0,
                signals={"reason": "invalid_dataframe"},
            )

        columns = {str(c).lower() for c in df.columns}
        matches = columns.intersection(self.RETAIL_COLUMNS)

        raw_score = len(matches) / len(self.RETAIL_COLUMNS)
        confidence = min(raw_score * 1.5, 1.0)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals={
                "matched_columns": sorted(matches),
                "match_count": len(matches),
            },
        )
