from typing import Set
import pandas as pd

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


class OpsDomain(BaseDomain):
    name = "ops"
    description = "Operations analytics domain"

    def validate_data(self, df: pd.DataFrame) -> bool:
        required = {"cycle_time", "on_time"}
        return isinstance(df, pd.DataFrame) and required.issubset(df.columns)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------
    # REQUIRED ABSTRACT METHODS
    # -------------------------

    def calculate_kpis(self, df: pd.DataFrame):
        from sreejita.reporting.ops.kpis import compute_ops_kpis
        return compute_ops_kpis(df)

    def generate_insights(self, df: pd.DataFrame, kpis):
        from sreejita.reporting.ops.insights import generate_ops_insights
        return generate_ops_insights(df, kpis)


class OpsDomainDetector(BaseDomainDetector):
    domain_name = "ops"

    OPS_COLUMNS: Set[str] = {
        "cycle_time",
        "lead_time",
        "on_time",
        "delay",
        "throughput",
        "capacity",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("ops", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.OPS_COLUMNS)

        confidence = min(len(matches) / len(self.OPS_COLUMNS) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="ops",
            confidence=confidence,
            signals={"matched_columns": sorted(matches)},
        )


def register(registry):
    registry.register(
        name="ops",
        domain_cls=OpsDomain,
        detector_cls=OpsDomainDetector,
    )
