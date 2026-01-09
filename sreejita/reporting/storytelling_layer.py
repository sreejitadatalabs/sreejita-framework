# =====================================================
# CLIENT STORYTELLING LAYER
# Non-Analytical | Non-Invasive | Executive-Safe
# =====================================================

from typing import List, Dict, Any


# -----------------------------------------------------
# RULES: INFO → OPPORTUNITY (SAFE PROMOTION)
# -----------------------------------------------------

OPPORTUNITY_PATTERNS = {
    "product contribution": {
        "title": "Revenue Concentration Opportunity",
        "rewrite": (
            "Revenue is concentrated among a small set of products, "
            "creating an opportunity to accelerate growth through focused "
            "investment while actively managing concentration risk."
        ),
    },
    "order value": {
        "title": "Order Value Expansion Opportunity",
        "rewrite": (
            "Order value distribution indicates headroom for upsell and "
            "bundling strategies to increase revenue per transaction."
        ),
    },
    "sales trend": {
        "title": "Demand Momentum Opportunity",
        "rewrite": (
            "Observed sales trends provide an opportunity to reinforce "
            "successful demand drivers and scale winning patterns."
        ),
    },
}


# -----------------------------------------------------
# STRATEGIC NARRATIVE TEMPLATES (1–2 MAX)
# -----------------------------------------------------

def generate_strategic_narratives(
    kpis: Dict[str, Any],
    domain: str,
) -> List[Dict[str, Any]]:

    narratives: List[Dict[str, Any]] = []

    total_sales = kpis.get("sales_total_sales") or kpis.get("total_sales")
    aov = kpis.get("sales_aov") or kpis.get("aov")

    if total_sales and aov:
        narratives.append({
            "level": "OPPORTUNITY",
            "title": "Scaling Core Revenue Drivers",
            "so_what": (
                "Strong baseline revenue combined with measurable order value "
                "creates an opportunity to scale growth through targeted "
                "commercial initiatives."
            ),
            "confidence": 0.8,
        })

    if kpis.get("primary_sub_domain") == "sales":
        narratives.append({
            "level": "OPPORTUNITY",
            "title": "Commercial Focus Advantage",
            "so_what": (
                "Sales-led performance provides clarity on where leadership "
                "focus can deliver near-term commercial impact."
            ),
            "confidence": 0.75,
        })

    return narratives[:2]


# -----------------------------------------------------
# MAIN ENTRY: ENHANCE INSIGHTS
# -----------------------------------------------------

def apply_storytelling_layer(
    insights: List[Dict[str, Any]],
    kpis: Dict[str, Any],
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Enhances insights for client storytelling WITHOUT
    changing analytical truth.
    """

    enhanced: List[Dict[str, Any]] = []
    promoted = 0

    for ins in insights:
        if not isinstance(ins, dict):
            continue

        title = ins.get("title", "").lower()
        level = ins.get("level", "INFO")

        rewritten = False

        if level == "INFO":
            for key, rule in OPPORTUNITY_PATTERNS.items():
                if key in title and promoted < 2:
                    enhanced.append({
                        "level": "OPPORTUNITY",
                        "title": rule["title"],
                        "so_what": rule["rewrite"],
                        "confidence": ins.get("confidence", 0.75),
                        "sub_domain": ins.get("sub_domain"),
                    })
                    promoted += 1
                    rewritten = True
                    break

        if not rewritten:
            enhanced.append(ins)

    # Add strategic narratives (max 2)
    enhanced.extend(generate_strategic_narratives(kpis, domain))

    return enhanced
