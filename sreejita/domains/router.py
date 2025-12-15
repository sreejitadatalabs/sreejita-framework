from typing import List
from sreejita.domains.contracts import DomainDetectionResult
from sreejita.domains.retail import RetailDomainDetector
from sreejita.domains.customer import CustomerDomainDetector
from sreejita.domains.finance import FinanceDomainDetector
from sreejita.domains.ops import OpsDomainDetector
from sreejita.domains.healthcare import HealthcareDomainDetector


class DomainRouter:
    """
    Central brain for domain detection.
    """

    def __init__(self):
        self.detectors = [
            RetailDomainDetector(),
            CustomerDomainDetector(),
            FinanceDomainDetector(),
            OpsDomainDetector(),
            HealthcareDomainDetector(),
        ]

    def detect_domains(self, df) -> List[DomainDetectionResult]:
        results = []

        for detector in self.detectors:
            result = detector.detect(df)
            if result.confidence > 0:
                results.append(result)

        # sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
