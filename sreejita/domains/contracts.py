# =====================================================
# DOMAIN CONTRACTS — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# =====================================================
# BASE DOMAIN DETECTOR
# =====================================================

class BaseDomainDetector:
    """
    Base class for all domain detectors.

    CONTRACT (v3.6):
    - detect(df) MUST return DomainDetectionResult
    - Detector MUST NOT attach engine
    - Detector MUST NOT raise (router guards, but detectors should be safe)
    """

    domain_name: str = "generic"

    def detect(self, df):
        raise NotImplementedError(
            "Domain detectors must implement detect(df)"
        )


# =====================================================
# DOMAIN DETECTION RESULT (AUTHORITATIVE)
# =====================================================

@dataclass
class DomainDetectionResult:
    """
    Domain Detection Result — Authoritative Contract

    This object represents:
    - WHY a domain matched (signals)
    - HOW confident the detector is

    Execution engine is attached LATER by the router.
    """

    domain: Optional[str]
    confidence: float
    signals: Dict[str, Any] = field(default_factory=dict)

    # Execution handle (router-attached, NEVER detector-attached)
    engine: Optional[Any] = None

    # -------------------------------------------------
    # POST-INIT NORMALIZATION (CRITICAL)
    # -------------------------------------------------
    def __post_init__(self):
        # Normalize domain
        if not isinstance(self.domain, str) or not self.domain:
            self.domain = None

        # Normalize confidence
        try:
            conf = float(self.confidence)
        except Exception:
            conf = 0.0

        # Clamp confidence to [0.0, 1.0]
        self.confidence = max(0.0, min(conf, 1.0))

        # Normalize signals
        if not isinstance(self.signals, dict):
            self.signals = {}

        # Engine must NEVER be set by detector
        # Router owns this lifecycle
        if self.engine is not None:
            # Defensive: strip silently
            self.engine = None
