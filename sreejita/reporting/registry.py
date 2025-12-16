from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations

DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations
    }
}
