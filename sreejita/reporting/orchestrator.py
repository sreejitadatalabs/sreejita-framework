from pathlib import Path
from datetime import datetime

from sreejita.reporting.registry import (
    DOMAIN_REPORT_ENGINES,
    DOMAIN_VISUALS,
    DOMAIN_NARRATIVES,
)


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain

    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    # -------------------------
    # KPIs (Phase 2.2 – Step 1)
    # -------------------------
    from sreejita.reporting.kpi_engine import normalize_kpis

    try:
        raw_kpis = engine["kpis"](df)
    except Exception as e:
        raw_kpis = {}

    if not raw_kpis:
        raw_kpis = {}

    kpis = normalize_kpis(raw_kpis)

    # -------------------------
    # INSIGHTS (Phase 2.2 – Step 2)
    # -------------------------
    insights_fn = engine.get("insights")
    insights = []

    if insights_fn:
        try:
            raw_insights = insights_fn(df, kpis) or []
        except TypeError:
            raw_insights = insights_fn(df) or []

        # ✅ Semantic validation added here
        from sreejita.reporting.insights import normalize_and_validate_insights
        insights = normalize_and_validate_insights(raw_insights)

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    recs_fn = engine.get("recommendations")
    if recs_fn:
        try:
            recommendations = recs_fn(df, kpis, insights)
        except TypeError:
            recommendations = recs_fn(df)
    else:
        recommendations = []

    # -------------------------
    # VISUALS
    # -------------------------
    visuals = []
    visual_hooks = DOMAIN_VISUALS.get(domain, {}).get("__always__", [])

    output_dir = Path("hybrid_images")
    output_dir.mkdir(exist_ok=True)

    for hook in visual_hooks:
        path = hook(df, output_dir)
        if path:
            visuals.append({
                "path": path,
                "caption": hook.__doc__ or ""
            })

    # -------------------------
    # NARRATIVE
    # -------------------------
    narrative_fn = DOMAIN_NARRATIVES.get(domain)
    narrative = narrative_fn() if narrative_fn else {}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals,
        "policy": policy.status,
        "narrative": narrative,
    }
    
