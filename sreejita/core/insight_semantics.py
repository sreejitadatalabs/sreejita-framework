LEVEL_TONE = {
    "GOOD": "positive",
    "WARNING": "cautionary",
    "RISK": "negative",
}


def validate_insight_semantics(insight: dict):
    """
    Ensures insight wording matches its severity level.
    """
    level = insight.get("level")
    tone = LEVEL_TONE.get(level)

    if not tone:
        return insight

    text = (
        insight.get("why", "") +
        insight.get("so_what", "")
    ).lower()

    if level == "GOOD" and any(w in text for w in ["risk", "decline", "problem"]):
        insight["semantic_warning"] = "Positive level with negative wording"

    if level == "RISK" and any(w in text for w in ["strong", "healthy", "good"]):
        insight["semantic_warning"] = "Negative level with positive wording"

    return insight
