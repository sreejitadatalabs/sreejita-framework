from datetime import datetime
from pathlib import Path

from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.reporting.generic_visuals import generate_generic_visuals


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
                "level": "RISK",
                "title": "Domain Resolution Failure",
                "so_what": "No analytical domain could be applied."
            }],
            "recommendations": [{
                "action": "Validate dataset structure",
                "rationale": "Domain identification failed",
            }],
            "visuals": generate_generic_visuals(df, output_dir),
            "policy": getattr(policy, "status", "UNKNOWN"),
        }

    # KPIs
    kpis = engine.calculate_kpis(df) or {}

    # Insights
    insights = engine.generate_insights(df, kpis) or []

    if not insights:
        insights = [{
            "level": "INFO",
            "title": "Baseline Data Review",
            "so_what": "Dataset processed successfully with no threshold breaches."
        }]

    # Recommendations
    raw_recs = engine.generate_recommendations(df, kpis) or []
    recommendations = enrich_recommendations(raw_recs)

    if not recommendations:
        recommendations = enrich_recommendations([{
            "action": "Review top-performing metrics",
            "rationale": "Identify opportunities for scaling strengths",
        }])

    # Visuals
    visuals = engine.generate_visuals(df, output_dir) or []
    if len(visuals) < 4:
        visuals.extend(
            generate_generic_visuals(df, output_dir, 4 - len(visuals))
        )

    # Risks (progressive)
    risks = []
    if policy.status != "allowed":
        risks.append({
            "level": "WARNING",
            "title": "Policy Constraint",
            "description": policy.status,
        })

    if not risks:
        risks.append({
            "level": "LOW",
            "title": "Operational Risk",
            "description": "No immediate risks detected based on available data."
        })

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals[:4],
        "risks": risks,
        "policy": policy.status,
    }
