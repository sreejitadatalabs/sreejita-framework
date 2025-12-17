from pathlib import Path
from datetime import datetime
from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES, DOMAIN_VISUALS


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain

    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    # -------------------------
    # KPIs
    # -------------------------
    kpis = engine["kpis"](df)

    # -------------------------
    # INSIGHTS
    # -------------------------
    insights_fn = engine.get("insights")
    insights = []

    if insights_fn:
        try:
            insights = insights_fn(df, kpis) or []
        except TypeError:
            insights = insights_fn(df) or []

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
    # VISUALS (ADAPTER SAFE)
    # -------------------------
    visuals = []
    visual_hooks = DOMAIN_VISUALS.get(domain, {}).get("__always__", [])

    output_dir = Path("hybrid_images")
    output_dir.mkdir(exist_ok=True)

    for hook in visual_hooks:
        try:
            path = hook(df, output_dir)
        except Exception:
            # HARD SAFETY: visuals must never break report
            continue

        if path:
            visuals.append(
                {
                    "path": path,
                    "caption": hook.__doc__ or "",
                }
            )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals,
        "policy": policy.status,
    }
