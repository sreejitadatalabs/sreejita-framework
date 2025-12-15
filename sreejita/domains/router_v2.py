"""
Router v2.0

Registry-based routing.
Does NOT import domain modules directly.
"""

from typing import Optional

from sreejita.domains.registry import registry
from sreejita.domains.contracts import DomainDetectionResult


def detect_domain(df) -> Optional[DomainDetectionResult]:
    """
    Detect domain using registry detectors.
    """
    return registry.detect_domain(df)


def apply_domain(df):
    """
    Apply domain preprocessing using detected domain.
    """

    result = detect_domain(df)

    if result is None or result.domain is None:
        # v1.x safety: no detection → return df unchanged
        return df

    domain = registry.get_domain(result.domain)

    if domain is None:
        # Domain not registered (safe fallback)
        return df

    # Apply domain-specific preprocessing
    try:
        return domain.preprocess(df)
    except Exception:
        # Domain logic must never crash pipeline
        return df
