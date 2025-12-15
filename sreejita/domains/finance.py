class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "Finance"

    FINANCE_COLUMNS = {
        "amount", "balance", "profit",
        "loss", "expense", "income",
        "transaction_id", "tax"
    }

    def detect(self, df):
        columns = set(c.lower() for c in df.columns)
        matches = columns.intersection(self.FINANCE_COLUMNS)

        score = len(matches) / len(self.FINANCE_COLUMNS)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=min(score * 1.6, 1.0),
            signals={"matched_columns": list(matches)}
        )
