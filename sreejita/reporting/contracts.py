"""
Reporting Contracts
-------------------
Authoritative output contracts for reporting layer.

Rules:
- Domain engines may return partial objects
- Reporting layer MUST normalize them
- Contracts here are domain-agnostic
"""

from typing import Dict, Any, List
from copy import deepcopy


# =====================================================
# RECOMMENDATION OUTPUT CONTRACT
# =====================================================

RECOMMENDATION_FIELDS: Dict[str, Any] = {
    "action": "",                    # What to do (required for display)
    "priority": "MEDIUM",             # CRITICAL | HIGH | MEDIUM | LOW
    "expected_outcome": "",           # Business / operational outcome
    "timeline": "TBD",                # e.g. 30 days, Q2, Immediate
    "owner": "Business Team",          # Accountable owner
    "confidence": None,               # 0–1 (optional, numeric)
    "rationale": "",                  # Why this recommendation exists
}


def normalize_recommendation(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single recommendation to contract.

    - Missing fields are filled
    - Extra fields are preserved (forward compatible)
    - Never raises
    """
    base = deepcopy(RECOMMENDATION_FIELDS)

    if not isinstance(rec, dict):
        return base

    for k, v in rec.items():
        base[k] = v

    # Hard safety rules
    if not base["action"]:
        base["action"] = "Review operational performance"

    if base["priority"]:
        base["priority"] = str(base["priority"]).upper()
    else:
        base["priority"] = "MEDIUM"

    if base["confidence"] is not None:
        try:
            base["confidence"] = round(float(base["confidence"]), 2)
        except Exception:
            base["confidence"] = None

    return base


def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize a list of recommendations.

    Guarantees:
    - Always returns a list
    - Each item matches RECOMMENDATION_FIELDS
    """
    if not isinstance(recommendations, list):
        return []

    normalized = []
    for rec in recommendations:
        try:
            normalized.append(normalize_recommendation(rec))
        except Exception:
            continue

    return normalized


# =====================================================
# INSIGHT OUTPUT CONTRACT (OPTIONAL BUT FUTURE-SAFE)
# =====================================================

INSIGHT_FIELDS: Dict[str, Any] = {
    "level": "INFO",                  # INFO | RISK | WARNING | CRITICAL
    "title": "",
    "so_what": "",
    "source": "",
    "executive_summary_flag": False,
}


def normalize_insight(insight: Dict[str, Any]) -> Dict[str, Any]:
    base = deepcopy(INSIGHT_FIELDS)

    if not isinstance(insight, dict):
        return base

    for k, v in insight.items():
        base[k] = v

    if not base["title"]:
        base["title"] = "Operational Observation"

    if base["level"]:
        base["level"] = str(base["level"]).upper()
    else:
        base["level"] = "INFO"

    return base


def normalize_insights(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(insights, list):
        return []

    return [normalize_insight(i) for i in insights]


# =====================================================
# VISUAL OUTPUT CONTRACT (REFERENCE)
# =====================================================

VISUAL_FIELDS: Dict[str, Any] = {
    "path": "",
    "caption": "",
    "importance": 0.0,   # 0–1 (used for ranking)
}


def normalize_visual(vis: Dict[str, Any]) -> Dict[str, Any]:
    base = deepcopy(VISUAL_FIELDS)

    if not isinstance(vis, dict):
        return base

    for k, v in vis.items():
        base[k] = v

    try:
        base["importance"] = float(base["importance"])
    except Exception:
        base["importance"] = 0.0

    return base
