class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "Customer"

    CUSTOMER_COLUMNS = {
        "customer_id", "customer_name", "email",
        "age", "gender", "location",
        "signup_date", "churn", "segment"
    }

    def detect(self, df):
        columns = set(c.lower() for c in df.columns)
        matches = columns.intersection(self.CUSTOMER_COLUMNS)

        score = len(matches) / len(self.CUSTOMER_COLUMNS)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=min(score * 1.4, 1.0),
            signals={"matched_columns": list(matches)}
        )
