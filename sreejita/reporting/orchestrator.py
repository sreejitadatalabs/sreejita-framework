from datetime import datetime
from pathlib import Path

from sreejita.reporting.generic_visuals import generate_generic_visuals
from sreejita.reporting.recommendation_enricher import enrich_recommendations


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain
    engine = getattr(decision, "engine", None)
    output_dir = Path("reports") / "visuals"

    if engine is None:
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "domain": domain,
            "kpis": {},
            "insights": [{
                "level": "CRITICAL",
                "title": "Domain Resolution Failed",
                "so_what": "No suitable analytical domain could be applied."
            }],
            "recommendations": [],
            "visuals": [],
            "risks": [{
                "level": "CRITICAL",
                "description": "Analysis could not be completed."
            }],
            "policy": policy.status,
        }

    # KPIs
    kpis = engine.calculate_kpis(df) or {}

    # Domain insights
    insights = engine.generate_insights(df, kpis) or []

    # Domain recommendations ONLY
    raw_recs = engine.generate_recommendations(df, kpis) or []
    recommendations = enrich_recommendations(raw_recs) if raw_recs else []

    # Visuals (domain first)
    visuals = engine.generate_visuals(df, output_dir) or []

    # Generic visuals ONLY if domain visuals < 2
    if len(visuals) < 2:
        visuals.extend(
            generate_generic_visuals(df, output_dir, max_visuals=2 - len(visuals))
        )

    # Risks (progressive)
    risks = []

    if policy.status != "allowed":
        risks.append({
            "level": "WARNING",
            "description": f"Policy status: {policy.status}"
        })

    for i in insights:
        if i.get("level") == "RISK":
            risks.append({
                "level": "CRITICAL",
                "description": i.get("title")
            })

    if not risks:
        risks.append({
            "level": "LOW",
            "description": "No material operational risks identified."
        })

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals,
        "risks": risks,
        "policy": policy.status,
    }
