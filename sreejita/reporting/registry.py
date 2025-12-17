# =====================================================
# REPORT ENGINE REGISTRY — v2.8.2
# =====================================================

# -------------------------
# RETAIL (v2.8 — THRESHOLD-BASED)
# -------------------------
from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations

# -------------------------
# CUSTOMER
# -------------------------
from sreejita.reporting.customer.kpis import compute_customer_kpis
from sreejita.reporting.customer.insights import generate_customer_insights
from sreejita.reporting.customer.recommendations import generate_customer_recommendations

# -------------------------
# FINANCE
# -------------------------
from sreejita.reporting.finance.kpis import compute_finance_kpis
from sreejita.reporting.finance.insights import generate_finance_insights
from sreejita.reporting.finance.recommendations import generate_finance_recommendations

# -------------------------
# OPS
# -------------------------
from sreejita.reporting.ops.kpis import compute_ops_kpis
from sreejita.reporting.ops.insights import generate_ops_insights
from sreejita.reporting.ops.recommendations import generate_ops_recommendations

# -------------------------
# HEALTHCARE
# -------------------------
from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis
from sreejita.reporting.healthcare.insights import generate_healthcare_insights
from sreejita.reporting.healthcare.recommendations import generate_healthcare_recommendations


# =====================================================
# DOMAIN → ENGINE MAP
# =====================================================

DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations,
    },
    "customer": {
        "kpis": compute_customer_kpis,
        "insights": generate_customer_insights,
        "recommendations": generate_customer_recommendations,
    },
    "finance": {
        "kpis": compute_finance_kpis,
        "insights": generate_finance_insights,
        "recommendations": generate_finance_recommendations,
    },
    "ops": {
        "kpis": compute_ops_kpis,
        "insights": generate_ops_insights,
        "recommendations": generate_ops_recommendations,
    },
    "healthcare": {
        "kpis": compute_healthcare_kpis,
        "insights": generate_healthcare_insights,
        "recommendations": generate_healthcare_recommendations,
    },
}

# =====================================================
# VISUAL REGISTRY — v2.8.2 (POLISHED)
# =====================================================

from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import heatmap, shipping_cost_vs_sales


DOMAIN_VISUALS = {
    "retail": {
        "__always__": [
            plot_monthly,             # WHAT happened (trend)
            bar,                      # WHERE happened (category)
            shipping_cost_vs_sales,   # WHY happened (cost vs value)
            heatmap,                  # RELATIONSHIPS (correlation)
        ]
    }
}
