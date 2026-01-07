"""
Router v2.1 — Universal, Detector-Authoritative, Hint-Aware

Sreejita Framework v3.6.1

RULES:
- Router NEVER guesses domains
- Router NEVER imports domain modules
- Router RESPECTS detector confidence
- Router HONORS user hint ONLY if domain exists
- Router NEVER forces healthcare
"""

from typing import Optional
import pandas as pd

from sreejita.domains.registry import registry
from sreejita.domains.contracts import DomainDetectionResult


# =====================================================
# GOVERNANCE CONSTANTS
# =====================================================

MIN_CONFIDENCE_ACCEPT = 0.40     # below this → reject
HINT_CONFIDENCE = 0.95           # user-selected trust
FALLBACK_CONFIDENCE = 0.35       # informational only


# =====================================================
# DOMAIN DETECTION — AUTHORITATIVE
# =====================================================

def detect_domain(
    df: pd.DataFrame,
    *,
    domain_hint: Optional[str] = None,
    strict: bool = False,
) -> DomainDetectionResult:
    """
    Canonical domain detection.

    Priority order:
    1. User hint (if valid & registered)
    2. Detector-based scoring
    3. Safe fallback (UNKNOWN)

    Returns DomainDetectionResult ALWAYS.
    """

    # -------------------------------------------------
    # SAFETY: INVALID INPUT
    # -------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return DomainDetectionResult(
            domain=None,
            confidence=0.0,
            signals={"error": "empty_or_invalid_dataframe"},
        )

    # -------------------------------------------------
    # STEP 1: USER DOMAIN HINT (TRUSTED BUT VERIFIED)
    # -------------------------------------------------
    if isinstance(domain_hint, str) and domain_hint.strip():
        hint = domain_hint.strip().lower()

        if registry.has_domain(hint):
            return DomainDetectionResult(
                domain=hint,
                confidence=HINT_CONFIDENCE,
                signals={"user_hint": hint},
            )
        else:
            # Invalid hint — record but do NOT force
            hint_signal = {"invalid_user_hint": hint}
    else:
        hint_signal = {}

    # -------------------------------------------------
    # STEP 2: DETECTOR-BASED RESOLUTION (PRIMARY PATH)
    # -------------------------------------------------
    best: Optional[DomainDetectionResult] = None

    for domain_name in registry.list_domains():
        detector = registry.get_detector(domain_name)
        if detector is None:
            continue

        try:
            result = detector.detect(df)

            if not isinstance(result, DomainDetectionResult):
                continue

            # Normalize engine ownership (router only)
            result.engine = None

            if best is None or result.confidence > best.confidence:
                best = result

        except Exception:
            # Detectors must never crash router
            continue

    # -------------------------------------------------
    # STEP 3: CONFIDENCE GOVERNANCE
    # -------------------------------------------------
    if best and best.domain and best.confidence >= MIN_CONFIDENCE_ACCEPT:
        return best

    if strict:
        # Strict mode → reject weak detections
        return DomainDetectionResult(
            domain=None,
            confidence=best.confidence if best else 0.0,
            signals={
                "reason": "strict_mode_reject",
                "best_candidate": best.domain if best else None,
                "best_confidence": best.confidence if best else 0.0,
                **hint_signal,
            },
        )

    # -------------------------------------------------
    # STEP 4: SAFE FALLBACK (INFORMATIONAL ONLY)
    # -------------------------------------------------
    return DomainDetectionResult(
        domain=None,
        confidence=best.confidence if best else FALLBACK_CONFIDENCE,
        signals={
            "reason": "no_confident_domain_detected",
            "best_candidate": best.domain if best else None,
            "best_confidence": best.confidence if best else 0.0,
            **hint_signal,
        },
    )


# =====================================================
# LEGACY HELPER — PREPROCESS ONLY (SAFE)
# =====================================================

def apply_domain(df: pd.DataFrame, *, domain_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Applies domain preprocessing ONLY if detection is confident.

    NEVER forces domain logic.
    NEVER crashes pipeline.
    """

    result = detect_domain(df, domain_hint=domain_hint)

    if not result or not result.domain or result.confidence < MIN_CONFIDENCE_ACCEPT:
        return df

    domain = registry.get_domain(result.domain)
    if domain is None:
        return df

    try:
        return domain.preprocess(df)
    except Exception:
        return df
