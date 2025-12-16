from pathlib import Path
from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES, DOMAIN_VISUALS


def generate_report_payload(df, decision, policy):
    domain = decision.selected_domain
    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    # -------------------------
    # KPIs / Insights / Recs
    # -------------------------
    kpis = engine["kpis"](df)
    insights = engine["insights"](df, kpis)
    recommendations = engine["recommendations"](insights)

    if not kpis:
        insights = []
        recommendations = []

    # -------------------------
    # Visual generation
    # -------------------------
    visuals = []
    visual_dir = Path("hybrid_images").resolve()
    visual_dir.mkdir(exist_ok=True)

    visual_map = DOMAIN_VISUALS.get(domain, {})

    # -------------------------------------------------
    # MINIMUM VISUAL SET (ALWAYS)
    # -------------------------------------------------
    if "__always__" in visual_map:
        captions = [
            "Sales trend over time (What happened)",
            "Sales contribution by category (Where it happened)",
            "Shipping cost vs sales relationship (Why it happened)",
        ]

        for visual_fn, caption in zip(visual_map["__always__"], captions):
            try:
                img_path = visual_fn(df, visual_dir)
                if img_path:
                    visuals.append({
                        "path": img_path,
                        "caption": caption
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
