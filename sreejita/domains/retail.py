"""
Retail Domain Module
Contains BOTH:
- RetailDomain (analytics)
- RetailDomainDetector (routing)
"""

from typing import Dict, Any, List, Set
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# ---------- Retail Analytics ----------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "profit" in df.columns and "sales" in df.columns:
        df["margin"] = df["profit"] / df["sales"]
    return df


def domain_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    if "sales" in df.columns:
        out["Total Sales"] = df["sales"].sum()
    if "profit" in df.columns:
        out["Total Profit"] = df["profit"].sum()
    return out


class RetailDomain(BaseDomain):
    name = "retail"
    description = "Retail analytics"
    required_columns = ["sales"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "sales" in df.columns or "revenue" in df.columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return enrich(df)

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        return domain_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        insights = []
        if "Total Sales" in kpis:
            insights.append(f"Total Sales: ${kpis['Total Sales']:,.0f}")
        if "Total Profit" in kpis:
            insights.append(f"Total Profit: ${kpis['Total Profit']:,.0f}")
        return insights


# ---------- Retail Detector ----------

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    RETAIL_COLUMNS: Set[str] = {
        "sales", "revenue", "profit", "discount",
        "product", "category", "sub_category",
        "quantity", "price", "order_id"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("retail", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.RETAIL_COLUMNS)

        score = min((len(matches) / len(self.RETAIL_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="retail",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )

# v2.0 registration hook
def register(registry):
    registry.register(
        name="retail",
        domain_cls=RetailDomain,
        detector_cls=RetailDomainDetector,
    )
