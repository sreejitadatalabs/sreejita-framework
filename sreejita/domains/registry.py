"""
Domain Registry (v2.0)

Single source of truth for:
- Domain implementations
- Domain detectors
- Domain resolution logic

v1.x compatible
v2.x extensible
"""

from typing import Dict, Type, List, Optional
from threading import Lock

from sreejita.domains.contracts import DomainDetectionResult, BaseDomainDetector


class DomainRegistry:
    """
    Central registry for all domains and detectors.

    This class is intentionally:
    - deterministic
    - explicit
    - side-effect free
    """

    def __init__(self):
        self._domains: Dict[str, object] = {}
        self._detectors: List[BaseDomainDetector] = []
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        domain_cls: Type,
        detector_cls: Type[BaseDomainDetector],
    ) -> None:
        """
        Register a domain + its detector.

        This must be called exactly once per domain.
        """

        name = name.lower().strip()

        with self._lock:
            if name in self._domains:
                raise ValueError(f"Domain '{name}' is already registered")

            detector = detector_cls()
            if not isinstance(detector, BaseDomainDetector):
                raise TypeError("detector_cls must inherit BaseDomainDetector")

            self._domains[name] = domain_cls()
            self._detectors.append(detector)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_domain(self, name: str):
        """Return domain implementation by name."""
        return self._domains.get(name)

    def get_domains(self) -> Dict[str, object]:
        """Return all registered domains."""
        return dict(self._domains)

    def get_detectors(self) -> List[BaseDomainDetector]:
        """Return all registered detectors."""
        return list(self._detectors)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_domain(self, df) -> Optional[DomainDetectionResult]:
        """
        Run all detectors and return the highest-confidence result.
        """

        best: Optional[DomainDetectionResult] = None

        for detector in self._detectors:
            try:
                result = detector.detect(df)
            except Exception:
                # v1.x rule: detectors must NEVER crash routing
                continue

            if not isinstance(result, DomainDetectionResult):
                continue

            if best is None or result.confidence > best.confidence:
                best = result

        return best


# ----------------------------------------------------------------------
# Singleton (v2.0 standard)
# ----------------------------------------------------------------------

registry = DomainRegistry()
