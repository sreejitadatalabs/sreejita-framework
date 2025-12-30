"""
Recommendation Enricher
-----------------------
Normalizes domain recommendations into a stable reporting contract.

Rules:
- Never crash
- Never drop recommendations
- Never allow missing required fields
- Preserve domain-added fields (forward compatible)
"""

from typing import List, Dict, Any
from copy import deepcopy

from sreejita.reporting.contracts import (
    RECOMMENDATION_FIELDS,
    normalize_recommendation,
)


def enrich_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich and normalize recommendations from any domain.

    Guarantees:
    - Always returns a list
    - Each recommendation conforms to reporting contract
    - Confidence, priority, and action are always present
    """

    if not isinstance(recommendations, list):
        return []

    enriched: List[Dict[str, Any]] = []

    for rec in recommendations:
        try:
            # Primary normalization (authoritative)
            normalized = normalize_recommendation(rec)

            # Backward compatibility:
            # Some domains may still emit `expected_impact`
            if "expected_impact" in normalized and not normalized.get("expected_outcome"):
                normalized["expected_outcome"] = normalized.pop("expected_impact")

            enriched.append(normalized)

        except Exception:
            # Absolute safety fallback
            fallback = deepcopy(RECOMMENDATION_FIELDS)
            fallback["action"] = "Review operational performance"
            fallback["priority"] = "MEDIUM"
            enriched.append(fallback)

    return enriched
