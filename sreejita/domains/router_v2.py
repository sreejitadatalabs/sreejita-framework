"""
Router v2.0

Registry-based routing.
Does NOT import domain modules directly.
"""

from typing import Optional

from sreejita.domains.registry import registry
from sreejita.domains.contracts import DomainDetectionResult


def detect_domain(df, domain_hint=None, strict=False):
    """
    Detect domain with optional hints and non-strict fallback.
    """
    # If user provides domain hint, use that
    if domain_hint and domain_hint.lower() == "healthcare":
        return "healthcare"
    
    # Try detection
    detected = domain_detector.detect(df)
    
    # If detection fails and we have minimal healthcare signals, use healthcare
    if detected.confidence < 0.40:
        healthcare_signals = {
            c.lower() for c in df.columns
        } & healthcare_keywords
        if len(healthcare_signals) >= 2:
            return "healthcare"  # Safe fallback

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
