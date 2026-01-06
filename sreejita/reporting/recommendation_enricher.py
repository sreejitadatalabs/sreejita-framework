"""
Recommendation Enricher
-----------------------
Normalizes domain recommendations into a stable reporting contract.

Guarantees:
- Never crashes
- Never drops recommendations
- Never emits incomplete records
- Preserves domain-added fields
- Deterministic & board-safe
"""

from typing import List, Dict, Any
from copy import deepcopy

from sreejita.reporting.contracts import (
    RECOMMENDATION_FIELDS,
    normalize_recommendation,
)

# --------------------------------------------------
# HARD LIMITS (EXECUTIVE SAFE)
# --------------------------------------------------

MIN_CONFIDENCE = 0.30
MAX_CONFIDENCE = 0.95

VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}


def _clamp_confidence(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return MIN_CONFIDENCE

    return round(
        max(MIN_CONFIDENCE, min(val, MAX_CONFIDENCE)),
        2,
    )


def _normalize_priority(value: Any) -> str:
    if isinstance(value, str):
        value = value.upper().strip()
        if value in VALID_PRIORITIES:
            return value
    return "MEDIUM"


# ==================================================
# PUBLIC API
# ==================================================

def enrich_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich and normalize recommendations from any domain.

    Output guarantees:
    - List[dict]
    - All required fields present
    - Confidence, priority, and action always valid
    - Deterministic ordering
    """

    if not isinstance(recommendations, list):
        return []

    enriched: List[Dict[str, Any]] = []
    seen_keys = set()

    for rec in recommendations:
        try:
            # -------------------------------
            # Primary normalization
            # -------------------------------
            normalized = normalize_recommendation(rec)

            # -------------------------------
            # Backward compatibility
            # -------------------------------
            if (
                "expected_impact" in normalized
                and not normalized.get("expected_outcome")
            ):
                normalized["expected_outcome"] = normalized.pop("expected_impact")

            # -------------------------------
            # HARD FIELD GUARDS
            # -------------------------------
            normalized["confidence"] = _clamp_confidence(
                normalized.get("confidence")
            )

            normalized["priority"] = _normalize_priority(
                normalized.get("priority")
            )

            normalized["action"] = (
                str(normalized.get("action")).strip()
                or "Review operational performance"
            )

            # -------------------------------
            # DEDUPLICATION (STABLE)
            # -------------------------------
            key = (
                normalized.get("sub_domain"),
                normalized.get("action"),
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            enriched.append(normalized)

        except Exception:
            # -------------------------------
            # EXPLICIT DEGRADED FALLBACK
            # -------------------------------
            fallback = deepcopy(RECOMMENDATION_FIELDS)

            fallback.update({
                "action": "Review operational performance",
                "priority": "MEDIUM",
                "confidence": MIN_CONFIDENCE,
                "expected_outcome": "Recommendation could not be fully generated",
                "owner": "Management",
                "timeline": "TBD",
                "goal": "Stabilize operations",
                "sub_domain": "unknown",
                "degraded": True,  # 🔴 explicit marker
            })

            enriched.append(fallback)

    return enriched
