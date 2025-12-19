from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

MIN_CONFIDENCE_FLOOR = 0.30
RULE_DOMINANCE_THRESHOLD = 0.70


def compute_domain_scores(df, rule_based_results):
    normalized_cols, mapping = normalize_columns(df.columns)

    final_scores = {}

    for domain, rb in rule_based_results.items():
        rule_conf = rb.get("confidence", 0)

        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        intent_conf = min(intent_score / 20.0, 1.0)

        # -----------------------------
        # 🔒 DOMINANCE PRESERVATION
        # -----------------------------
        if rule_conf >= RULE_DOMINANCE_THRESHOLD:
            # Intent can only boost, not override
            combined = max(rule_conf, round(rule_conf + (0.15 * intent_conf), 3))
        else:
            # Normal weighted blend
            combined = round(
                (0.6 * rule_conf) + (0.4 * intent_conf),
                3
            )

        final_scores[domain] = {
            "confidence": combined,
            "rule_confidence": rule_conf,
            "intent_confidence": intent_conf,
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