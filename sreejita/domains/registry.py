"""
Domain Registry — Universal (FINAL)
Sreejita Framework v3.5.x

This module defines the SINGLE authoritative registry
used across the entire framework.

Design principles:
- Deterministic
- Explicit registration
- No dynamic discovery
- Safe for orchestration & explainability
"""

from typing import Dict, Type, Optional, List


class DomainRegistry:
    """
    Central registry for domain implementations and detectors.

    Responsibilities:
    - Maintain domain → implementation mapping
    - Maintain domain → detector mapping
    - Provide safe accessors (never crash)
    """

    def __init__(self):
        self._domains: Dict[str, Type] = {}
        self._detectors: Dict[str, Type] = {}

    # -------------------------------------------------
    # REGISTRATION
    # -------------------------------------------------

    def register(
        self,
        name: str,
        domain_cls: Type,
        detector_cls: Optional[Type] = None,
    ) -> None:
        """
        Register a domain implementation (and optional detector).

        Parameters:
        - name: canonical domain name (e.g. "healthcare")
        - domain_cls: BaseDomain subclass
        - detector_cls: BaseDomainDetector subclass (optional)
        """

        if not isinstance(name, str) or not name:
            raise ValueError("Domain name must be a non-empty string")

        self._domains[name] = domain_cls

        if detector_cls is not None:
            self._detectors[name] = detector_cls

    # -------------------------------------------------
    # ACCESSORS
    # -------------------------------------------------

    def get_domain(self, name: str):
        """
        Return a NEW instance of a domain implementation.

        Returns:
        - Domain instance
        - None if domain not registered
        """
        domain_cls = self._domains.get(name)
        if not domain_cls:
            return None

        try:
            return domain_cls()
        except Exception:
            # Absolute safety: registry must never crash execution
            return None

    def get_detector(self, name: str):
        """
        Return a NEW instance of a domain detector.

        Returns:
        - Detector instance
        - None if detector not registered
        """
        detector_cls = self._detectors.get(name)
        if not detector_cls:
            return None

        try:
            return detector_cls()
        except Exception:
            return None

    # -------------------------------------------------
    # INTROSPECTION
    # -------------------------------------------------

    def list_domains(self) -> List[str]:
        """
        List all registered domain names.
        """
        return sorted(self._domains.keys())

    def list_detectors(self) -> List[str]:
        """
        List all domains that have detectors.
        """
        return sorted(self._detectors.keys())

    def has_domain(self, name: str) -> bool:
        return name in self._domains

    def has_detector(self, name: str) -> bool:
        return name in self._detectors


# =====================================================
# 🔒 SINGLETON REGISTRY (AUTHORITATIVE)
# =====================================================

registry = DomainRegistry()
