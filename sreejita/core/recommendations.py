def generate_prescriptive_recommendations(summary_recommendations):
    """
    v1.9.9
    Converts executive summary recommendations into structured,
    prescriptive action blocks.
    Fully rule-based and domain-agnostic.
    """

    expanded = []

    archetypes = [
        {
            "priority": "High",
            "outcome": "Reduce downside risk and stabilize performance."
        },
        {
            "priority": "Medium",
            "outcome": "Improve efficiency and unlock incremental gains."
        },
        {
            "priority": "Medium",
            "outcome": "Strengthen governance and decision reliability."
        }
    ]

    for idx, rec in enumerate(summary_recommendations):
        meta = archetypes[idx % len(archetypes)]

        expanded.append({
            "action": rec,
            "rationale": (
                "This recommendation directly addresses a pattern observed "
                "in the analysis and aligns with standard operational controls."
            ),
            "expected_outcome": meta["outcome"],
            "priority": meta["priority"],
        })

    return expanded
