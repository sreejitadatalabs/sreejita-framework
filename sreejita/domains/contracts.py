from dataclasses import dataclass
from typing import Dict, Any


class BaseDomainDetector:
    """Base class for domain detectors (v1.x public API)."""
    domain_name: str = "generic"

    def detect(self, df):
        raise NotImplementedError


@dataclass
class DomainDetectionResult:
    domain: str
    confidence: float
    signals: Dict[str, Any]
