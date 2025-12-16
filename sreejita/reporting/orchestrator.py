from pathlib import Path
from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES, DOMAIN_VISUALS


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain
    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    kpis = engine["kpis"](df)
    insights = engine["insights"](df, kpis)
    recommendations = engine["recommendations"](insights)

    visuals = []
    visual_dir = Path("hybrid_images").resolve()
    visual_dir.mkdir(exist_ok=True)

    visual_map = DOMAIN_VISUALS.get(domain, {})

    # -------------------------
    # Insight-driven visuals
    # -------------------------
    for ins in insights:
        metric = ins.get("metric")
        if metric and metric in visual_map:
            try:
                img_path = visual_map[metric](df, visual_dir)
                visuals.append({
                    "path": img_path,
                    "caption": ins["title"]
                })
            except Exception:
                pass

    # -------------------------
    # BASELINE visual fallback
    # -------------------------
    if not visuals and "__baseline__" in visual_map:
        try:
            img_path = visual_map["__baseline__"](df, visual_dir)
            visuals.append({
                "path": img_path,
                "caption": "Baseline distribution — no significant risks detected"
            })
        except Exception:
            pass

    return {
        "domain": domain,
        "domain_confidence": decision.confidence,
        "policy_status": policy.status if policy else None,
        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations,
        "visuals": visuals,
    }
