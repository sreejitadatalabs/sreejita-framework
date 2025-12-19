# sreejita/domains/intelligence/intent_scoring.py

from .domain_intents import DOMAIN_INTENTS

HIGH_WEIGHT = 3
AMBIGUOUS_WEIGHT = 1


def score_domain_intent(normalized_columns, domain: str):
    """
    Returns:
        score: int
        signals: dict
    """

    intents = DOMAIN_INTENTS.get(domain)
    if not intents:
        return 0, {}

    high_hits = intents["high"].intersection(normalized_columns)
    amb_hits = intents["ambiguous"].intersection(normalized_columns)

    score = (
        len(high_hits) * HIGH_WEIGHT
        + len(amb_hits) * AMBIGUOUS_WEIGHT
    )

    # =====================================================
    # 🔑 HR vs CUSTOMER DOMINANCE RULE (MANDATORY)
    # =====================================================
    if domain == "customer":
        HR_EXCLUSIVE_SIGNALS = {
            "salary", "compensation", "ctc", "payroll",
            "attrition", "termination", "resignation",
            "performance", "rating",
            "leave", "absence", "attendance"
        }

        if HR_EXCLUSIVE_SIGNALS.intersection(normalized_columns):
            # Penalize customer intent if HR signals exist
            score = int(score * 0.5)

    return score, {
        "high_confidence_matches": sorted(high_hits),
        "ambiguous_matches": sorted(amb_hits),
    }
