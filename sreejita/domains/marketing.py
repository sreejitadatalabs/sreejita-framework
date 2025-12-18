from typing import Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class MarketingDomain(BaseDomain):
    name = "marketing"
    description = "Marketing analytics domain"

    def validate_data(self, df: pd.DataFrame) -> bool:
        required = {"campaign_id", "converted", "cost"}
        return isinstance(df, pd.DataFrame) and required.issubset(df.columns)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------
    # REQUIRED ABSTRACT METHODS
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame):
        from sreejita.reporting.marketing.kpis import compute_marketing_kpis
        return compute_marketing_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis):
        from sreejita.reporting.marketing.insights import generate_marketing_insights
        return generate_marketing_insights(df, kpis)


class MarketingDomainDetector(BaseDomainDetector):
    domain_name = "marketing"

    MARKETING_COLUMNS: Set[str] = {
        "campaign",
        "campaign_id",
        "converted",
        "clicks",
        "impressions",
        "cost",
        "spend",
        "ctr",
        "conversion_rate",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("marketing", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.MARKETING_COLUMNS)

        confidence = min(len(matches) / len(self.MARKETING_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="marketing",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


def register(registry):
    registry.register(
        name="marketing",
        domain_cls=MarketingDomain,
        detector_cls=MarketingDomainDetector,
    )
