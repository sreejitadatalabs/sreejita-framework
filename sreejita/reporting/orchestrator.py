from pathlib import Path
from datetime import datetime
from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES, DOMAIN_VISUALS


def generate_report_payload(df, decision, policy, config):
    """
    Generate a domain-specific report payload (v2.8+)
    Fully config-driven, deterministic
    """

    domain = decision.selected_domain
    engine = DOMAIN_REPORT_ENGINES.get(domain)

    if not engine:
        raise RuntimeError(f"No report engine registered for domain '{domain}'")

    # -------------------------
    # KPIs (MANDATORY)
    # -------------------------
    kpis = engine["kpis"](df, config)

    # -------------------------
    # INSIGHTS
    # -------------------------
    insights_fn = engine.get("insights")
    insights = []

    if insights_fn:
        try:
            insights = insights_fn(df, kpis, config) or []
        except TypeError:
            insights = insights_fn(df, kpis) or []

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    recs_fn = engine.get("recommendations")
    recommendations = []

    if recs_fn:
        try:
            recommendations = recs_fn(df, kpis, insights, config)
        except TypeError:
            recommendations = recs_fn(df, kpis, insights)

    # -------------------------
    # VISUALS (OPTIONAL)
    # -------------------------
    visuals = []
    visual_hooks = DOMAIN_VISUALS.get(domain, {}).get("__always__", [])

    output_dir = Path("hybrid_images")
    output_dir.mkdir(exist_ok=True)

    for hook in visual_hooks:
        path = hook(df, output_dir, config)
        if path:
            visuals.append({
                "path": path,
                "caption": hook.__doc__ or ""
            })

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals,
        "policy": policy.status,
    }
