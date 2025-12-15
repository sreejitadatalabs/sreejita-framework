from typing import Any, Dict, Set

from sreejita.domains.contracts import (
    BaseDomainDetector,
    DomainDetectionResult,
)


class RetailDomainDetector(BaseDomainDetector):
    """
    Domain detector for Retail datasets.

    This class is part of the public v1.x domain interface.
    DO NOT rename or remove until v2.0.
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
        """
        Detect whether the dataset belongs to the Retail domain.

        Detection is based on column-name signals only.
        This method must NEVER raise.
        """

        # Defensive: empty or invalid dataframe
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult(
                domain=self.domain_name,
                confidence=0.0,
                signals={"reason": "invalid_dataframe"},
            )

        columns = {str(c).lower() for c in df.columns}

        matches = columns.intersection(self.RETAIL_COLUMNS)

        # Simple normalized score
        raw_score = len(matches) / len(self.RETAIL_COLUMNS)

        # Slight boost but capped (deterministic, explainable)
        confidence = min(raw_score * 1.5, 1.0)

        signals: Dict[str, Any] = {
            "matched_columns": sorted(matches),
            "match_count": len(matches),
            "total_signals": len(self.RETAIL_COLUMNS),
        }

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals=signals,
        )
