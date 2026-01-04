# =====================================================
# SUB-DOMAIN ENGINE — UNIVERSAL
# Sreejita Framework v3.5+
# =====================================================
"""
Universal Sub-Domain Inference Engine

Responsibilities:
- Detect semantic signals safely
- Score sub-domains deterministically
- Guarantee UNKNOWN fallback
- Remain domain-agnostic

This engine MUST:
- Never crash
- Never return empty results
- Never assume schema
"""

from typing import Dict, Optional, Callable
import pandas as pd


# =====================================================
# SAFE SIGNAL DETECTION (CANONICAL)
# =====================================================
def has_signal(df: pd.DataFrame, col: Optional[str]) -> bool:
    """
    Returns True if:
    - Column name exists
    - Column is present in DataFrame
    - At least one non-null value exists

    This is the ONLY approved signal checker
    across the entire framework.
    """
    return bool(
        col
        and isinstance(col, str)
        and col in df.columns
        and df[col].notna().any()
    )


# =====================================================
# SUB-DOMAIN ENGINE
# =====================================================
class SubDomainEngine:
    """
    Universal engine for sub-domain inference.

    Usage:
        engine = SubDomainEngine("healthcare")
        scores = engine.infer(df, cols, rules)

    Rules are passed in by domains.
    """

    def __init__(self, domain_name: str):
        self.domain_name = domain_name

    # -------------------------------------------------
    # CORE INFERENCE ENTRY POINT
    # -------------------------------------------------
    def infer(
        self,
        df: pd.DataFrame,
        cols: Dict[str, Optional[str]],
        rules: Dict[str, Callable[[pd.DataFrame, Dict[str, Optional[str]]], float]],
        unknown_label: str = "unknown",
        min_confidence: float = 0.3,
    ) -> Dict[str, float]:
        """
        Infer sub-domain confidence scores.

        Args:
            df: input dataset
            cols: resolved semantic columns
            rules: dict[sub_domain -> scoring function]
            unknown_label: fallback sub-domain
            min_confidence: activation threshold

        Returns:
            Dict[sub_domain -> confidence_score]
        """

        scores: Dict[str, float] = {}

        # -------------------------------
        # APPLY RULES (DETERMINISTIC)
        # -------------------------------
        for sub_domain, rule_fn in rules.items():
            try:
                score = rule_fn(df, cols)

                if isinstance(score, (int, float)) and score > 0:
                    scores[sub_domain] = round(float(score), 2)

            except Exception:
                # Absolute safety: ignore faulty rules
                continue

        # -------------------------------
        # FILTER WEAK SIGNALS
        # -------------------------------
        active = {
            k: v for k, v in scores.items()
            if v >= min_confidence
        }

        # -------------------------------
        # HARD FALLBACK — UNKNOWN
        # -------------------------------
        if not active:
            return {unknown_label: 1.0}

        return active


# =====================================================
# RULE BUILDING HELPERS (OPTIONAL, REUSABLE)
# =====================================================
def rule_any(*checks: Callable[[pd.DataFrame, Dict[str, Optional[str]]], bool], score: float = 0.8):
    """
    Rule fires if ANY check returns True.
    """
    def _rule(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> float:
        return score if any(check(df, cols) for check in checks) else 0.0
    return _rule


def rule_all(*checks: Callable[[pd.DataFrame, Dict[str, Optional[str]]], bool], score: float = 0.9):
    """
    Rule fires if ALL checks return True.
    """
    def _rule(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> float:
        return score if all(check(df, cols) for check in checks) else 0.0
    return _rule


# =====================================================
# COMMON CHECK BUILDERS
# =====================================================
def has_col(col_key: str):
    """
    Returns a check function for column existence.
    """
    def _check(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> bool:
        return has_signal(df, cols.get(col_key))
    return _check


def lacks_col(col_key: str):
    """
    Returns a check function ensuring column absence.
    """
    def _check(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> bool:
        return not has_signal(df, cols.get(col_key))
    return _check
