from datetime import datetime
from pathlib import Path


def generate_report_payload(df, decision, policy):
    """
    v2.x HARDENED ORCHESTRATOR
    - NEVER returns None
    - NEVER crashes Hybrid
    """

    domain = decision.selected_domain
    engine = getattr(decision, "engine", None)

    # Absolute safety net
    if engine is None:
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "domain": domain,
            "kpis": {},
            "insights": [{
                "level": "RISK",
                "title": "Domain Engine Missing",
                "so_what": "No domain engine was resolved for this dataset."
            }],
            "recommendations": [],
            "visuals": [],
            "narrative": {},
            "policy": getattr(policy, "status", "UNKNOWN"),
        }

    try:
        # KPIs
        kpis = (
            engine.calculate_kpis(df)
            if hasattr(engine, "calculate_kpis")
            else {}
        ) or {}

        # Insights
        insights = (
            engine.generate_insights(df, kpis)
            if hasattr(engine, "generate_insights")
            else []
        ) or []

        # Recommendations
        recommendations = (
            engine.generate_recommendations(df, kpis)
            if hasattr(engine, "generate_recommendations")
            else []
        ) or []

        # Visuals
        output_dir = Path("reports") / "visuals"
        visuals = (
            engine.generate_visuals(df, output_dir)
            if hasattr(engine, "generate_visuals")
            else []
        ) or []

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "domain": domain,
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
            "narrative": {},
            "policy": getattr(policy, "status", "UNKNOWN"),
        }

    except Exception as e:
        # LAST LINE OF DEFENSE — NEVER BREAK HYBRID
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "domain": domain,
            "kpis": {},
            "insights": [{
                "level": "RISK",
                "title": "Analysis Execution Failure",
                "so_what": str(e),
            }],
            "recommendations": [],
            "visuals": [],
            "narrative": {},
            "policy": getattr(policy, "status", "UNKNOWN"),
        }
