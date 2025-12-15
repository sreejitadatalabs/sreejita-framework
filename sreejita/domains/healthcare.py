class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "Healthcare"

    HEALTH_COLUMNS = {
        "patient_id", "diagnosis", "treatment",
        "admission_date", "discharge_date",
        "doctor", "hospital", "cost"
    }

    def detect(self, df):
        columns = set(c.lower() for c in df.columns)
        matches = columns.intersection(self.HEALTH_COLUMNS)

        score = len(matches) / len(self.HEALTH_COLUMNS)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=min(score * 1.7, 1.0),
            signals={"matched_columns": list(matches)}
        )
