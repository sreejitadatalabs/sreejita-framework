"""
Recommendation Enricher
-----------------------
Normalizes domain recommendations into a stable reporting contract.

GUARANTEES:
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


# ==================================================
# INTERNAL HELPERS (SAFE & PURE)
# ==================================================

def _clamp_confidence(value: Any) -> float:
    """
    Clamp confidence into board-safe range.
    """
    try:
        val = float(value)
    except Exception:
        return MIN_CONFIDENCE

    return round(
        max(MIN_CONFIDENCE, min(val, MAX_CONFIDENCE)),
        2,
    )


def _normalize_priority(value: Any) -> str:
    """
    Normalize priority into controlled enum.
    """
    if isinstance(value, str):
        value = value.upper().strip()
        if value in VALID_PRIORITIES:
            return value
    return "MEDIUM"


def _safe_str(value: Any, default: str) -> str:
    try:
        s = str(value).strip()
        return s if s else default
    except Exception:
        return default


# ==================================================
# PUBLIC API (AUTHORITATIVE)
# ==================================================

def enrich_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich and normalize recommendations from any domain.

    OUTPUT GUARANTEES:
    - List[Dict[str, Any]]
    - All required fields present
    - Confidence always bounded
    - Priority normalized
    - Stable deduplication
    - Deterministic ordering
    """

    if not isinstance(recommendations, list):
        return []

    enriched: List[Dict[str, Any]] = []
    seen_keys = set()

    for rec in recommendations:
        try:
            # -------------------------------------------------
            # 1. PRIMARY NORMALIZATION (AUTHORITATIVE CONTRACT)
            # -------------------------------------------------
            normalized = normalize_recommendation(rec)

            # -------------------------------------------------
            # 2. BACKWARD COMPATIBILITY
            # -------------------------------------------------
            if (
                "expected_impact" in normalized
                and not normalized.get("expected_outcome")
            ):
                normalized["expected_outcome"] = normalized.pop(
                    "expected_impact"
                )

            # -------------------------------------------------
            # 3. HARD FIELD GUARDS (NON-NEGOTIABLE)
            # -------------------------------------------------
            normalized["confidence"] = _clamp_confidence(
                normalized.get("confidence")
            )

            normalized["priority"] = _normalize_priority(
                normalized.get("priority")
            )

            normalized["action"] = _safe_str(
                normalized.get("action"),
                "Review operational performance",
            )

            normalized["owner"] = _safe_str(
                normalized.get("owner"),
                "Management",
            )

            normalized["timeline"] = _safe_str(
                normalized.get("timeline"),
                "TBD",
            )

            normalized["goal"] = _safe_str(
                normalized.get("goal"),
                "Stabilize operations",
            )

            normalized["sub_domain"] = _safe_str(
                normalized.get("sub_domain"),
                "unknown",
            )

            # -------------------------------------------------
            # 4. STABLE DEDUPLICATION (SUBDOMAIN + ACTION)
            # -------------------------------------------------
            dedup_key = (
                normalized.get("sub_domain"),
                normalized.get("action"),
            )

            if dedup_key in seen_keys:
                continue

            seen_keys.add(dedup_key)
            enriched.append(normalized)

        except Exception:
            # -------------------------------------------------
            # 5. EXPLICIT DEGRADED FALLBACK (NEVER DROP)
            # -------------------------------------------------
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

    # -------------------------------------------------
    # 6. DETERMINISTIC EXECUTIVE SORTING
    # -------------------------------------------------
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    enriched.sort(
        key=lambda r: (
            priority_rank.get(r.get("priority"), 3),
            -float(r.get("confidence", 0.0)),
            r.get("action", ""),
        )
    )

    return enriched
