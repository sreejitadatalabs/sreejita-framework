# Ecommerce reporting module
from .kpis import compute_ecommerce_kpis
from .insights import generate_ecommerce_insights
from .recommendations import generate_ecommerce_recommendations
from .narrative import get_domain_narrative

__all__ = [
    'compute_ecommerce_kpis',
    'generate_ecommerce_insights',
    'generate_ecommerce_recommendations',
    'get_domain_narrative'
]
