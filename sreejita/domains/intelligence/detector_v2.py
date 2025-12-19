from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

MIN_CONFIDENCE_FLOOR = 0.30
MAX_CONFIDENCE_CAP = 1.0


def compute_domain_scores(df, rule_based_results):
    """
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
        dict domain -> {
            confidence,
            rule_confidence,
            intent_confidence,
            signals
        }
    """

    normalized_cols, mapping = normalize_columns(df.columns)
    final_scores = {}

    for domain, rb in rule_based_results.items():
        rule_conf = rb.get("confidence", 0.0)

        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        # Normalize intent score to 0–1
        intent_conf = min(intent_score / 20.0, 1.0)

        # Combine scores
        combined = (0.6 * rule_conf) + (0.4 * intent_conf)

        # 🔒 HARD SAFETY CAP (THIS WAS MISSING)
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


def select_best_domain(domain_scores):
    if not domain_scores:
        return "unknown", 0.0, {}

    domain, meta = max(
        domain_scores.items(),
        key=lambda x: x[1]["confidence"]
    )

    if meta["confidence"] < MIN_CONFIDENCE_FLOOR:
        return "unknown", meta["confidence"], meta

    return domain, meta["confidence"], meta
