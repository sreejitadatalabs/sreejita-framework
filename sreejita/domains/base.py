from abc import ABC
from dataclasses import dataclass
from typing import Dict, Set


# -----------------------------
# Signal Weights (Global Contract)
# -----------------------------
SIGNAL_WEIGHTS = {
    "PRIMARY": 5.0,
    "SECONDARY": 2.0,
    "GENERIC": 0.5,
}


# -----------------------------
# Detection Result
# -----------------------------
@dataclass
class DomainDetectionResult:
    domain: str
    confidence: float
    signals: Dict[str, Set[str]]


# -----------------------------
# Base Detector
# -----------------------------
class BaseDomainDetector(ABC):
    """
    Base class for all domain detectors.

    Domains MUST define:
    - domain (str)
    - PRIMARY_SIGNALS
    - SECONDARY_SIGNALS
    - GENERIC_SIGNALS
    """

    domain: str

    PRIMARY_SIGNALS: Set[str] = set()
    SECONDARY_SIGNALS: Set[str] = set()
    GENERIC_SIGNALS: Set[str] = set()

    def detect(self, df) -> DomainDetectionResult:
        columns = {c.lower() for c in df.columns}

        primary_matches = columns & self.PRIMARY_SIGNALS
        secondary_matches = columns & self.SECONDARY_SIGNALS
        generic_matches = columns & self.GENERIC_SIGNALS

        raw_score = (
            SIGNAL_WEIGHTS["PRIMARY"] * len(primary_matches)
            + SIGNAL_WEIGHTS["SECONDARY"] * len(secondary_matches)
            + SIGNAL_WEIGHTS["GENERIC"] * len(generic_matches)
        )

        # Normalize to 0–1 range (safe cap)
        confidence = min(raw_score / 10.0, 1.0)

        return DomainDetectionResult(
            domain=self.domain,
            confidence=confidence,
            signals={
                "primary": primary_matches,
                "secondary": secondary_matches,
                "generic": generic_matches,
            },
        )