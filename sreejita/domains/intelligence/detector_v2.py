# =====================================================
# DOMAIN INTELLIGENCE — DETECTOR v2 (STABILIZED)
# Sreejita Framework v3.6
# =====================================================

from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

# -------------------------------------------------
# SAFETY CONSTANTS
# -------------------------------------------------

MIN_CONFIDENCE_FLOOR = 0.30     # below this → unknown
MAX_CONFIDENCE_CAP = 1.0

# Intent is SUPPORTING ONLY — never dominant
INTENT_WEIGHT = 0.25
RULE_WEIGHT = 0.75

# Intent normalization guard
INTENT_SCORE_MAX = 20.0


# -------------------------------------------------
# DOMAIN SCORING ENGINE (AUTHORITATIVE)
# -------------------------------------------------

def compute_domain_scores(df, rule_based_results):
    """
    Combine rule-based detection with intent signals.

    RULES:
    - Rule-based detection is authoritative
    - Intent can ONLY reinforce, never override
    - No new domains may be introduced by intent
    - Healthcare-safe by design
    """

    if not rule_based_results:
        return {}

    normalized_cols, _ = normalize_columns(df.columns)
    final_scores = {}

    for domain, rb in rule_based_results.items():

        rule_conf = float(rb.get("confidence", 0.0))
        rule_conf = max(0.0, min(rule_conf, 1.0))

        # 🚫 HARD GATE: no rule confidence → skip domain
        if rule_conf <= 0.0:
            continue

        # -------------------------------
        # INTENT SCORING (SUPPORTING ONLY)
        # -------------------------------
        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        # Normalize intent conservatively
        intent_conf = min(
            max(intent_score / INTENT_SCORE_MAX, 0.0),
            1.0,
        )

        # Penalize noisy intent
        if intent_conf < 0.15:
            intent_conf = 0.0

        # -------------------------------
        # COMBINED CONFIDENCE
        # -------------------------------
        combined = (
            RULE_WEIGHT * rule_conf +
            INTENT_WEIGHT * intent_conf
        )

        combined = round(
            min(MAX_CONFIDENCE_CAP, max(combined, 0.0)),
            3,
        )

        final_scores[domain] = {
            "confidence": combined,
            "rule_confidence": round(rule_conf, 3),
            "intent_confidence": round(intent_conf, 3),
            "signals": {
                "rule_based": rb.get("signals", {}),
                "intent_based": intent_signals or {},
            },
        }

    return final_scores


# -------------------------------------------------
# DOMAIN SELECTION LOGIC (CONSERVATIVE)
# -------------------------------------------------

def select_best_domain(domain_scores):
    """
    Select the highest-confidence domain.

    GUARANTEES:
    - Never forces a domain
    - UNKNOWN preferred over false positives
    """

    if not domain_scores:
        return "unknown", 0.0, {}

    domain, meta = max(
        domain_scores.items(),
        key=lambda x: x[1].get("confidence", 0.0),
    )

    confidence = meta.get("confidence", 0.0)

    # 🚫 HARD FLOOR
    if confidence < MIN_CONFIDENCE_FLOOR:
        return "unknown", confidence, meta

    return domain, confidence, meta
