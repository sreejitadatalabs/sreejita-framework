from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations
from sreejita.reporting.retail.visuals import _shipping_cost_vs_sales_v26

from sreejita.reporting.retail.visuals import (
    shipping_cost_vs_sales,
    discount_distribution,
)

DOMAIN_VISUALS = {
    "retail": {
        "shipping_cost_ratio": _shipping_cost_vs_sales_v26,
        "average_discount": discount_distribution,
    }
}


DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations
    }
}
