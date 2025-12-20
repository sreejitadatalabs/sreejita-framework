from dataclasses import dataclass
from typing import Dict, Any, Optional


class BaseDomainDetector:
    """
    Base class for domain detectors.

    v3.0 Contract:
    - detect() MUST return DomainDetectionResult
    - engine must be attached for execution
    """

    domain_name: str = "generic"

    def detect(self, df):
        raise NotImplementedError("Domain detectors must implement detect()")


@dataclass
class DomainDetectionResult:
    """
    v3.0 Domain Detection Contract
    --------------------------------
    This object is BOTH:
    - an explanation (why this domain)
    - an execution handle (how to analyze it)
    """

    domain: str
    confidence: float
    signals: Dict[str, Any]

    # 🔑 THIS IS THE MOST IMPORTANT FIX
    engine: Optional[Any] = None
