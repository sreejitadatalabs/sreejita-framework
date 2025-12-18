from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Set


# Signal weights (global, consistent)
SIGNAL_WEIGHTS = {
    "PRIMARY": 5.0,
    "SECONDARY": 2.0,
    "GENERIC": 0.5,
}


@dataclass
class DomainDetectionResult:
    domain: str
    confidence: float
    signals: Dict[str, Set[str]]


class BaseDomainDetector(ABC):
    domain: str

    PRIMARY_SIGNALS: Set[str] = set()
    SECONDARY_SIGNALS: Set[str] = set()
    GENERIC_SIGNALS: Set[str] = set()

    def detect(self, df) -> DomainDetectionResult:
        columns = {c.lower() for c in df.columns}

        primary = columns & self.PRIMARY_SIGNALS
        secondary = columns & self.SECONDARY_SIGNALS
        generic = columns & self.GENERIC_SIGNALS

        raw_score = (
            SIGNAL_WEIGHTS["PRIMARY"] * len(primary)
            + SIGNAL_WEIGHTS["SECONDARY"] * len(secondary)
            + SIGNAL_WEIGHTS["GENERIC"] * len(generic)
        )

        # Normalize to 0–1 (cap for safety)
        confidence = min(raw_score / 10.0, 1.0)

        return DomainDetectionResult(
            domain=self.domain,
            confidence=confidence,
            signals={
                "primary": primary,
                "secondary": secondary,
                "generic": generic,
            },
        )
