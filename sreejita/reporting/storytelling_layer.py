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

# =====================================================
# PHASE 2 — COMPARATIVE / DELTA INSIGHTS
# =====================================================

def generate_comparative_insights(
    df,
    kpis: Dict[str, Any],
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Generates contrast-based insights using existing data only.
    No new KPIs. No recomputation.
    """

    insights: List[Dict[str, Any]] = []

    # -------------------------------
    # 1. Top vs Long-Tail Products
    # -------------------------------
    sales_col = kpis.get("_resolved_columns", {}).get("sales")
    product_col = kpis.get("_resolved_columns", {}).get("product")

    if sales_col and product_col and sales_col in df.columns:
        prod_sales = (
            df.groupby(product_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
        )

        if len(prod_sales) >= 10:
            top_n = max(1, int(len(prod_sales) * 0.2))
            top_share = prod_sales.iloc[:top_n].sum() / prod_sales.sum()

            insights.append({
                "level": "OPPORTUNITY",
                "title": "Top vs Long-Tail Revenue Concentration",
                "so_what": (
                    f"The top {top_n} products contribute approximately "
                    f"{top_share:.0%} of total revenue, indicating a "
                    "power-law dynamic that can be strategically leveraged "
                    "or actively diversified."
                ),
                "confidence": 0.8,
                "sub_domain": "sales",
            })

    # -------------------------------
    # 2. Dominant Category vs Rest
    # -------------------------------
    category_col = kpis.get("_resolved_columns", {}).get("category")

    if sales_col and category_col and category_col in df.columns:
        cat_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)

        if len(cat_sales) >= 3:
            top_cat = cat_sales.iloc[0]
            rest = cat_sales.iloc[1:].sum()

            if rest > 0:
                ratio = top_cat / rest

                insights.append({
                    "level": "OPPORTUNITY",
                    "title": "Category Dominance Signal",
                    "so_what": (
                        f"The leading category generates roughly "
                        f"{ratio:.1f}× the revenue of all other categories combined, "
                        "highlighting both focus strength and category concentration exposure."
                    ),
                    "confidence": 0.75,
                    "sub_domain": "merchandising",
                })

    # -------------------------------
    # 3. Sales Variability Signal
    # -------------------------------
    if sales_col and sales_col in df.columns:
        mean_sales = df[sales_col].mean()
        std_sales = df[sales_col].std()

        if mean_sales > 0:
            cv = std_sales / mean_sales

            if cv >= 0.5:
                descriptor = "high"
            elif cv >= 0.25:
                descriptor = "moderate"
            else:
                descriptor = "low"

            insights.append({
                "level": "INFO",
                "title": "Sales Variability Profile",
                "so_what": (
                    f"Sales variability is {descriptor} relative to average volume, "
                    "suggesting demand is influenced by commercial levers "
                    "such as pricing, promotion, or assortment mix."
                ),
                "confidence": 0.7,
                "sub_domain": "sales",
            })

    return insights

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
    # After Phase 1 enhancements
    enhanced.extend(
        generate_comparative_insights(df, kpis, domain)
    )

    return enhanced
