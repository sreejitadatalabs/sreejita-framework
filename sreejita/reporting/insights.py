from typing import List, Dict, Any

from sreejita.core.insight_semantics import validate_insight_semantics


def normalize_and_validate_insights(
    insights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validates insight semantic consistency.
    Adds semantic warnings if tone mismatches severity.
    Does NOT block execution.
    """
    validated = []

    for insight in insights:
        validated.append(validate_insight_semantics(insight))

    return validated
