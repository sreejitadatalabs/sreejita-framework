from .domain_intents import DOMAIN_INTENTS

HIGH_WEIGHT = 5
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

    return score, {
        "high_confidence_matches": sorted(high_hits),
        "ambiguous_matches": sorted(amb_hits),
    }
