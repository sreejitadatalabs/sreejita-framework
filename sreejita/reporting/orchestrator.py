from datetime import datetime
from pathlib import Path
from sreejita.reporting.recommendation_enricher import enrich_recommendations


def generate_report_payload(df, decision, policy):
    """
    v2.x FINAL ORCHESTRATOR (LOCKED)

    Guarantees:
    - Domain lifecycle is respected (preprocess → kpis → insights → visuals)
    - Healthcare engine MUST exist
    - No silent failures
    """

    domain = decision.selected_domain
    engine = getattr(decision, "engine", None)

    # =====================================================
    # HARD PROTECTION — HEALTHCARE MUST HAVE ENGINE
    # =====================================================
    if domain == "healthcare" and engine is None:
        raise RuntimeError(
            "Healthcare domain detected but engine not attached. "
            "Check dispatch_domain() and domain bootstrap."
        )

    # =====================================================
    # SOFT FALLBACK — OTHER DOMAINS
    # =====================================================
    if engine is None:
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "domain": domain,
            "kpis": {},
            "insights": [{
                "level": "RISK",
                "title": "Domain Engine Missing",
                "so_what": "No domain engine could be resolved for this dataset.",
            }],
            "recommendations": [],
            "visuals": [],
            "narrative": {},
            "policy": getattr(policy, "status", "UNKNOWN"),
        }

    try:
        # =====================================================
        # 🔥 DOMAIN PREPROCESS (THIS WAS MISSING)
        # =====================================================
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        # =====================================================
        # KPIs (DOMAIN-OWNED)
        # =====================================================
        kpis = (
            engine.calculate_kpis(df)
            if hasattr(engine, "calculate_kpis")
            else {}
        ) or {}

        # =====================================================
        # INSIGHTS (DOMAIN-OWNED)
        # =====================================================
        insights = (
            engine.generate_insights(df, kpis)
            if hasattr(engine, "generate_insights")
            else []
        ) or []

        # =====================================================
        # RECOMMENDATIONS (ENRICH ONLY)
        # =====================================================
        raw_recs = (
            engine.generate_recommendations(df, kpis)
            if hasattr(engine, "generate_recommendations")
            else []
        ) or []

        recommendations = enrich_recommendations(raw_recs)

        # =====================================================
        # VISUALS (DOMAIN-OWNED)
        # =====================================================
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
        # =====================================================
        # FINAL DEFENSE — REPORT MUST STILL RENDER
        # =====================================================
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
