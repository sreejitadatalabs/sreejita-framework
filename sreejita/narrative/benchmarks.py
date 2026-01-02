# sreejita/narrative/benchmarks.py

"""
UNIVERSAL EXECUTIVE BENCHMARKS & GUARDRAILS
==========================================

This file defines UNIVERSAL, DOMAIN-AGNOSTIC benchmarks
used only for EXECUTIVE NARRATIVE CONTEXT.

RULES:
- No domain-specific KPIs
- No scoring logic
- No hard dependencies
- Safe for all domains (Healthcare, Retail, Finance, HR, etc.)

This file answers:
→ “What generally looks good / risky at an executive level?”
"""

from typing import Dict, Any, Optional


# =====================================================
# 1. CAPABILITY-LEVEL NARRATIVE BENCHMARKS
# =====================================================

CAPABILITY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "VOLUME": {
        "low": "Low activity volume relative to historical norms.",
        "normal": "Activity volume within expected operating range.",
        "high": "Sustained high activity volume requiring capacity review.",
    },
    "TIME_FLOW": {
        "good": "Process cycle times are within acceptable ranges.",
        "warning": "Process delays indicate emerging flow constraints.",
        "critical": "Sustained delays pose operational risk.",
    },
    "COST": {
        "efficient": "Costs appear aligned with delivered value.",
        "warning": "Cost growth may outpace output or outcomes.",
        "critical": "Cost structure presents material financial risk.",
    },
    "QUALITY": {
        "stable": "Quality indicators are stable and controlled.",
        "warning": "Early signs of quality degradation detected.",
        "critical": "Quality performance requires immediate attention.",
    },
    "VARIANCE": {
        "low": "Performance variation is well-controlled.",
        "high": "Significant variance suggests standardization gaps.",
    },
    "ACCESS": {
        "adequate": "Access levels appear sufficient for demand.",
        "limited": "Access constraints may affect outcomes or experience.",
    },
}


# =====================================================
# 2. EXECUTIVE CONFIDENCE BANDS (UNIVERSAL)
# =====================================================

CONFIDENCE_BANDS = [
    (0.85, "HIGH", "🟢"),
    (0.70, "MEDIUM", "🟡"),
    (0.00, "LOW", "🔴"),
]


def classify_confidence(confidence: Optional[float]) -> Dict[str, Any]:
    """
    Converts a numeric confidence (0–1) into
    an executive-friendly label and icon.
    """
    if confidence is None:
        return {"label": "UNKNOWN", "icon": "⚪"}

    for threshold, label, icon in CONFIDENCE_BANDS:
        if confidence >= threshold:
            return {
                "label": label,
                "icon": icon,
                "value": round(confidence, 2),
            }

    return {"label": "LOW", "icon": "🔴", "value": round(confidence, 2)}


# =====================================================
# 3. GOVERNANCE GUARDRAILS (UNIVERSAL SAFETY)
# =====================================================

GOVERNANCE_LIMITS: Dict[str, Dict[str, Any]] = {
    "COST": {
        "soft_cap_multiplier": 1.3,
        "hard_cap_multiplier": 2.0,
        "source": "Executive financial governance heuristic",
    },
    "TIME_FLOW": {
        "soft_cap_multiplier": 1.5,
        "hard_cap_multiplier": 2.5,
        "source": "Operational resilience guideline",
    },
}


def apply_governance_cap(
    capability: str,
    baseline: Optional[float],
    observed: Optional[float],
) -> Optional[float]:
    """
    Applies a universal governance cap to prevent
    narrative distortion due to extreme values.
    """
    if baseline is None or observed is None:
        return observed

    limits = GOVERNANCE_LIMITS.get(capability)
    if not limits:
        return observed

    hard = limits.get("hard_cap_multiplier")
    if hard:
        return min(observed, baseline * hard)

    return observed


# =====================================================
# 4. SAFE ACCESS HELPERS (CANONICAL API)
# =====================================================

def get_capability_benchmark(capability: str) -> Dict[str, str]:
    """
    Returns narrative benchmark text for a capability.
    """
    return CAPABILITY_BENCHMARKS.get(capability, {})


def get_governance_source(capability: str) -> str:
    """
    Returns governance reference source for a capability.
    """
    return GOVERNANCE_LIMITS.get(capability, {}).get("source", "")


# =====================================================
# END OF FILE — UNIVERSAL EXECUTIVE CONTEXT LAYER
# =====================================================
