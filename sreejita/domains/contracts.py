from abc import ABC, abstractmethod
from typing import Dict, List, Any

class DomainDetectionResult:
    def __init__(
        self,
        domain: str,
        confidence: float,
        signals: Dict[str, Any]
    ):
        self.domain = domain
        self.confidence = round(confidence, 3)
        self.signals = signals  # explainability payload


class BaseDomainDetector(ABC):
    """
    Contract for all domain detectors.
    """

    domain_name: str

    @abstractmethod
    def detect(self, df) -> DomainDetectionResult:
        """
        Analyze dataset and return domain confidence + signals.
        """
        pass
