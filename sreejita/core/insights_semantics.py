"""
Insight semantic validation utilities.

This module ensures that the textual tone of an insight
matches its declared severity level (GOOD / WARNING / RISK).

It does NOT block execution.
It only annotates inconsistencies for visibility.
"""

LEVEL_TONE = {
    "GOOD": "positive",
    "WARNING": "cautionary",
    "RISK": "negative",
}


def validate_insight_semantics(insight: dict) -> dict:
    """
    Adds a semantic_warning field if insight wording
    contradicts its severity level.
    """
    if not isinstance(insight, dict):
        return insight

    level = insight.get("level")
    if level not in LEVEL_TONE:
        return insight

    text = (
        f"{insight.get('why', '')} "
        f"{insight.get('so_what', '')}"
    ).lower()

    # Negative wording keywords
    negative_words = [
        "risk", "decline", "drop", "loss",
        "problem", "issue", "concern",
        "pressure", "negative", "weak",
    ]

    # Positive wording keywords
    positive_words = [
        "strong", "healthy", "good", "positive",
        "improved", "growth", "stable",
        "efficient", "optimal",
    ]

    if level == "GOOD" and any(w in text for w in negative_words):
        insight["semantic_warning"] = (
            "GOOD insight contains negative wording"
        )

    if level == "RISK" and any(w in text for w in positive_words):
        insight["semantic_warning"] = (
            "RISK insight contains positive wording"
        )

    return insight
