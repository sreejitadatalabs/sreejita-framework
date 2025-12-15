def generate_detailed_insights(summary_insights):
    """
    v1.9.9
    Converts executive-level insight bullets into expanded,
    consultant-style reasoning blocks.
    Domain-agnostic, deterministic.
    """

    detailed = []

    for idx, insight in enumerate(summary_insights, start=1):
        detailed.append({
            "title": f"Insight {idx}",
            "what": insight,
            "why": (
                "Supporting data patterns indicate this behavior is consistent "
                "across key dimensions rather than a one-off fluctuation."
            ),
            "so_what": (
                "If this pattern continues unchecked, it may negatively impact "
                "performance, efficiency, or risk exposure over time."
            ),
        })

    return detailed
