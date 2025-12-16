from sreejita.reporting.customer.kpis import compute_customer_kpis
from sreejita.reporting.customer.insights import generate_customer_insights
from sreejita.reporting.customer.recommendations import generate_customer_recommendations

from sreejita.reporting.finance.kpis import compute_finance_kpis
from sreejita.reporting.finance.insights import generate_finance_insights
from sreejita.reporting.finance.recommendations import generate_finance_recommendations

from sreejita.reporting.ops.kpis import compute_ops_kpis
from sreejita.reporting.ops.insights import generate_ops_insights
from sreejita.reporting.ops.recommendations import generate_ops_recommendations

from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis
from sreejita.reporting.healthcare.insights import generate_healthcare_insights
from sreejita.reporting.healthcare.recommendations import generate_healthcare_recommendations


DOMAIN_REPORT_ENGINES = {
    "retail": DOMAIN_REPORT_ENGINES["retail"],

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
