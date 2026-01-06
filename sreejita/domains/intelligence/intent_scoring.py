# =====================================================
# DOMAIN INTENT SCORING — STABILIZED
# Sreejita Framework v3.6
# =====================================================

from .domain_intents import DOMAIN_INTENTS

# -------------------------------------------------
# WEIGHTS (CONSERVATIVE)
# -------------------------------------------------

HIGH_WEIGHT = 3          # strong semantic signal
AMBIGUOUS_WEIGHT = 0.5  # weak context only (not additive dominance)

# Hard cap to prevent intent inflation
MAX_INTENT_SCORE = 10


# -------------------------------------------------
# DOMAIN-EXCLUSIVE PENALTY MAP
# -------------------------------------------------

DOMAIN_EXCLUSIVE_SIGNALS = {
    "healthcare": {
        "revenue", "profit", "margin", "invoice",
        "sku", "inventory", "sales"
    },
    "pharmacy": {
        "admission", "discharge", "bed", "icu",
        "surgery", "ward"
    },
    "finance": {
        "patient", "diagnosis", "treatment",
        "clinical", "mortality"
    },
    "customer": {
        "salary", "ctc", "payroll", "attrition",
        "leave", "attendance", "performance"
    },
    "hr": {
        "customer", "order", "purchase", "cart",
        "checkout"
    },
}


# -------------------------------------------------
# INTENT SCORING FUNCTION
# -------------------------------------------------

def score_domain_intent(normalized_columns, domain: str):
    """
    Conservative intent scoring.

    GUARANTEES:
    - Intent NEVER introduces domains
    - Ambiguous signals are weak
    - Cross-domain contamination is penalized
    """

    intents = DOMAIN_INTENTS.get(domain)
    if not intents:
        return 0, {}

    high_hits = intents.get("high", set()).intersection(normalized_columns)
    amb_hits = intents.get("ambiguous", set()).intersection(normalized_columns)

    # -------------------------------
    # BASE SCORE
    # -------------------------------
    score = (
        len(high_hits) * HIGH_WEIGHT +
        len(amb_hits) * AMBIGUOUS_WEIGHT
    )

    # -------------------------------
    # EXCLUSIVE SIGNAL PENALTY
    # -------------------------------
    exclusive = DOMAIN_EXCLUSIVE_SIGNALS.get(domain)
    if exclusive and exclusive.intersection(normalized_columns):
        # Penalize strongly — intent conflict
        score *= 0.5

    # -------------------------------
    # HARD SAFETY CAPS
    # -------------------------------
    score = max(0.0, min(score, MAX_INTENT_SCORE))

    # -------------------------------
    # CLEAN SIGNAL PAYLOAD
    # -------------------------------
    signals = {
        "high_confidence_matches": sorted(high_hits),
        "ambiguous_matches": sorted(amb_hits),
    }

    return float(score), signals
