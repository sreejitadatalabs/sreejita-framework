from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "Retail"

    RETAIL_COLUMNS = {
        "sales", "revenue", "order_id", "product",
        "sku", "quantity", "discount", "store",
        "category", "price"
    }

    def detect(self, df):
        columns = set(c.lower() for c in df.columns)

        matches = columns.intersection(self.RETAIL_COLUMNS)
        score = len(matches) / len(self.RETAIL_COLUMNS)

        signals = {
            "matched_columns": list(matches),
            "match_count": len(matches),
        }

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=min(score * 1.5, 1.0),
            signals=signals
        )
