from sreejita.reporting.contracts import RECOMMENDATION_FIELDS

def enrich_recommendations(recommendations):
    """
    Ensures every recommendation conforms to the contract.
    Missing fields are filled with defaults.
    """
    enriched = []

    for rec in recommendations or []:
        full = RECOMMENDATION_FIELDS.copy()
        full.update(rec)  # domain values override defaults
        enriched.append(full)

    return enriched
