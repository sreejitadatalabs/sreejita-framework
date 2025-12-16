from pathlib import Path
from datetime import datetime
from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES, DOMAIN_VISUALS


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain

    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    # -------------------------
    # KPIs (MANDATORY)
    # -------------------------
    kpis = engine["kpis"](df)

    # -------------------------
    # INSIGHTS (FIXED)
    # -------------------------
    insights_fn = engine.get("insights")
    if insights_fn:
        try:
            # ✅ PASS KPIs INTO INSIGHTS
            insights = insights_fn(df, kpis)
        except TypeError:
            # Backward compatibility (v1 insights)
            insights = insights_fn(df)
    else:
        insights = []

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

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,          # 🔥 NOW POPULATED
        "recommendations": recommendations,
        "visuals": visuals,
        "policy": policy.status,
    }
