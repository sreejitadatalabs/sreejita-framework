def generate_composite_insights(kpis, ranked_dimensions):
    """
    Detects entities that are problematic across multiple dimensions.
    """
    insights = []

    for entity, dimensions in ranked_dimensions.items():
        if len(dimensions) >= 2:
            insights.append({
                "level": "WARNING",
                "title": f"{entity} is a High-Impact Risk Area",
                "so_what": f"{entity} ranks high in {', '.join(dimensions)}, indicating compounded risk."
            })

    return insights
