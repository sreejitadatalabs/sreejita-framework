from typing import Dict, Any


class SubDomainSignal:
    """
    Domain-agnostic sub-domain signal contract.
    """

    def __init__(
        self,
        name: str,
        confidence: float,
        evidence: Dict[str, Any] | None = None,
    ):
        self.name = name
        self.confidence = round(float(confidence), 3)
        self.evidence = evidence or {}
