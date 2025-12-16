# =====================================================
# REPORT ENGINES REGISTRY (v2.6)
# =====================================================

# -------------------------
# Retail (BASE — MUST BE FIRST)
# -------------------------
from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations

DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations,
    }
}

# -------------------------
# Customer
# -------------------------
from sreejita.reporting.customer.kpis import compute_customer_kpis
from sreejita.reporting.customer.insights import generate_customer_insights
from sreejita.reporting.customer.recommendations import generate_customer_recommendations

# -------------------------
# Finance
# -------------------------
from sreejita.reporting.finance.kpis import compute_finance_kpis
from sreejita.reporting.finance.insights import generate_finance_insights
from sreejita.reporting.finance.recommendations import generate_finance_recommendations

# -------------------------
# Ops
# -------------------------
from sreejita.reporting.ops.kpis import compute_ops_kpis
from sreejita.reporting.ops.insights import generate_ops_insights
from sreejita.reporting.ops.recommendations import generate_ops_recommendations

# -------------------------
# Healthcare
# -------------------------
from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis
from sreejita.reporting.healthcare.insights import generate_healthcare_insights
from sreejita.reporting.healthcare.recommendations import generate_healthcare_recommendations


# -------------------------
# EXTEND REPORT ENGINES
# -------------------------
DOMAIN_REPORT_ENGINES.update({
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
})


# =====================================================
# VISUALS REGISTRY (v2.6)
# =====================================================

# -------------------------
# Retail visuals (BASE)
# -------------------------
# =====================================================
# VISUALS REGISTRY (v2.7.1)
# =====================================================
from sreejita.reporting.retail.visuals import (
    _sales_trend_v27,
    _sales_by_category_v27,
    _shipping_cost_vs_sales_v27,
    _discount_distribution_v26,
    _baseline_sales_distribution_v27,
)

DOMAIN_VISUALS["retail"] = {
    "__always__": [
        _sales_trend_v27,
        _sales_by_category_v27,
        _shipping_cost_vs_sales_v27,
         _discount_distribution_v26,
        _baseline_sales_distribution_v27
    ]
}


# ---- other domains already registered earlier ----

# -------------------------
# Customer visuals
# -------------------------
from sreejita.reporting.customer.visuals import _churn_proxy_distribution_v26

# -------------------------
# Finance visuals
# -------------------------
from sreejita.reporting.finance.visuals import _expense_vs_revenue_v26

# -------------------------
# Ops visuals
# -------------------------
from sreejita.reporting.ops.visuals import _sla_breach_rate_plot_v26

# -------------------------
# Healthcare visuals
# -------------------------
from sreejita.reporting.healthcare.visuals import _readmission_rate_plot_v26


# -------------------------
# EXTEND VISUALS
# -------------------------
DOMAIN_VISUALS.update({
    "customer": {
        "churn_proxy_rate": _churn_proxy_distribution_v26,
    },
    "finance": {
        "expense_ratio": _expense_vs_revenue_v26,
    },
    "ops": {
        "sla_breach_rate": _sla_breach_rate_plot_v26,
    },
    "healthcare": {
        "readmission_rate": _readmission_rate_plot_v26,
    },
})
