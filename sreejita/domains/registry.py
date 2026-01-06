# =====================================================
# DOMAIN REGISTRY — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

from typing import Dict, Type, Optional, List
import logging

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector

log = logging.getLogger("sreejita.registry")


class DomainRegistry:
    """
    Central authoritative registry for domain implementations and detectors.

    GUARANTEES:
    - Deterministic registration
    - Explicit domain ownership
    - Safe instantiation
    - No dynamic discovery
    """

    def __init__(self):
        self._domains: Dict[str, Type[BaseDomain]] = {}
        self._detectors: Dict[str, Type[BaseDomainDetector]] = {}

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    # -------------------------------------------------
    # REGISTRATION (AUTHORITATIVE)
    # -------------------------------------------------

    def register(
        self,
        name: str,
        domain_cls: Type,
        detector_cls: Optional[Type] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a domain implementation and optional detector.

        Rules:
        - domain_cls MUST be BaseDomain subclass
        - detector_cls MUST be BaseDomainDetector subclass (if provided)
        - Classes must be instantiable with no args
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Domain name must be a non-empty string")

        key = self._normalize(name)

        if not overwrite and key in self._domains:
            raise RuntimeError(
                f"Domain '{key}' already registered. "
                "Use overwrite=True only if intentional."
            )

        # -------------------------------
        # DOMAIN VALIDATION
        # -------------------------------
        if not isinstance(domain_cls, type) or not issubclass(domain_cls, BaseDomain):
            raise TypeError(
                f"Domain '{key}' must be a BaseDomain subclass"
            )

        try:
            domain_cls()
        except Exception as e:
            raise TypeError(
                f"Domain '{key}' cannot be instantiated safely: {e}"
            )

        self._domains[key] = domain_cls

        # -------------------------------
        # DETECTOR VALIDATION (OPTIONAL)
        # -------------------------------
        if detector_cls is not None:
            if (
                not isinstance(detector_cls, type)
                or not issubclass(detector_cls, BaseDomainDetector)
            ):
                raise TypeError(
                    f"Detector for '{key}' must be a BaseDomainDetector subclass"
                )

            try:
                detector = detector_cls()
                if not hasattr(detector, "detect"):
                    raise AttributeError("Detector missing detect()")
            except Exception as e:
                raise TypeError(
                    f"Detector for '{key}' cannot be instantiated safely: {e}"
                )

            # Soft consistency warning
            declared = getattr(detector_cls, "domain_name", None)
            if declared and declared != key:
                log.warning(
                    f"Detector domain_name='{declared}' does not match registry key='{key}'"
                )

            self._detectors[key] = detector_cls

    # -------------------------------------------------
    # ACCESSORS (NEVER CRASH)
    # -------------------------------------------------

    def get_domain(self, name: str):
        key = self._normalize(name)
        domain_cls = self._domains.get(key)
        if not domain_cls:
            return None

        try:
            return domain_cls()
        except Exception:
            return None

    def get_detector(self, name: str):
        key = self._normalize(name)
        detector_cls = self._detectors.get(key)
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
        return sorted(self._domains.keys())

    def list_detectors(self) -> List[str]:
        return sorted(self._detectors.keys())

    def has_domain(self, name: str) -> bool:
        return self._normalize(name) in self._domains

    def has_detector(self, name: str) -> bool:
        return self._normalize(name) in self._detectors


# =====================================================
# 🔒 SINGLETON REGISTRY (AUTHORITATIVE)
# =====================================================

registry = DomainRegistry()
