class OpsDomainDetector(BaseDomainDetector):
    domain_name = "Operations"

    OPS_COLUMNS = {
        "process", "cycle_time", "downtime",
        "throughput", "defect", "capacity",
        "efficiency", "utilization"
    }

    def detect(self, df):
        columns = set(c.lower() for c in df.columns)
        matches = columns.intersection(self.OPS_COLUMNS)

        score = len(matches) / len(self.OPS_COLUMNS)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=min(score * 1.3, 1.0),
            signals={"matched_columns": list(matches)}
        )
