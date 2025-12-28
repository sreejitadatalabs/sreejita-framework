from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

# -------------------------------------------------
# CONFIDENCE BOUNDS & HEURISTICS
# -------------------------------------------------

MIN_CONFIDENCE_FLOOR = 0.30   # below this, domain is considered uncertain
MAX_CONFIDENCE_CAP = 1.0

# Intent scores are normalized against an expected upper bound.
# This is a heuristic, not a probabilistic guarantee.
INTENT_SCORE_MAX = 20.0


# -------------------------------------------------
# DOMAIN SCORING ENGINE (v2)
# -------------------------------------------------

def compute_domain_scores(df, rule_based_results):
    """
    Combine rule-based detection confidence with intent-based signals.

    Args:
        df: pandas DataFrame
        rule_based_results: dict
            {
              domain: {
                "confidence": float,
                "signals": dict
              }
            }

    Returns:
        dict:
            domain -> {
                confidence: float,
                rule_confidence: float,
                intent_confidence: float,
                signals: {
                    rule_based: dict,
                    intent_based: dict
                }
            }

    Notes:
        - Domains without rule-based activation are intentionally excluded.
        - Scores are deterministic and capped for safety.
    """

    normalized_cols, mapping = normalize_columns(df.columns)
    final_scores = {}

    for domain, rb in (rule_based_results or {}).items():
        rule_conf = float(rb.get("confidence", 0.0))

        # Intent scoring is based on semantic column matches
        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        # Normalize intent score to 0–1 using heuristic upper bound
        intent_conf = min(intent_score / INTENT_SCORE_MAX, 1.0)

        # Weighted combination (rule-based dominance)
        combined = (0.6 * rule_conf) + (0.4 * intent_conf)

        # Hard safety cap and rounding for stability
        combined = round(
            max(0.0, min(combined, MAX_CONFIDENCE_CAP)),
            3
        )

        final_scores[domain] = {
            "confidence": combined,
            "rule_confidence": round(rule_conf, 3),
            "intent_confidence": round(intent_conf, 3),
            "signals": {
                "rule_based": rb.get("signals", {}),
                "intent_based": intent_signals,
            },
        }

    return final_scores


# -------------------------------------------------
# DOMAIN SELECTION LOGIC
# -------------------------------------------------

def select_best_domain(domain_scores):
    """
    Select the highest-confidence domain.

    Returns:
        (domain_name, confidence, metadata)

    Notes:
        - If no domain exceeds the minimum confidence floor,
          'unknown' is returned to prevent false positives.
    """

    if not domain_scores:
        return "unknown", 0.0, {}

    domain, meta = max(
        domain_scores.items(),
        key=lambda x: x[1]["confidence"]
    )

    if meta["confidence"] < MIN_CONFIDENCE_FLOOR:
        return "unknown", meta["confidence"], meta

    return domain, meta["confidence"], meta
