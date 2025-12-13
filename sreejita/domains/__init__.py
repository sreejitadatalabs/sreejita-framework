"""Sreejita Framework - Domain Modules

Domain modules extend the core engine with specialized KPIs, insights,
and preprocessing for specific business contexts.

Available Domains:
- retail: Retail analytics, sales, inventory
- ecommerce: E-commerce, conversions, cart analysis
- customer: Customer segmentation, profiling, behavior
- text: NLP preprocessing, sentiment, topic extraction
- finance: Financial metrics, forecasting, risk
"""

from .retail import RetailDomain
from .ecommerce import EcommerceDomain
from .customer import CustomerDomain
from .text import TextDomain
from .finance import FinanceDomain
from .base import BaseDomain

__all__ = [
    "RetailDomain",
    "EcommerceDomain",
    "CustomerDomain",
    "TextDomain",
    "FinanceDomain",
    "BaseDomain",
]

DOMAIN_REGISTRY = {
    "retail": RetailDomain,
    "ecommerce": EcommerceDomain,
    "customer": CustomerDomain,
    "text": TextDomain,
    "finance": FinanceDomain,
}

def get_domain(domain_name):
    """Dynamically load domain module by name."""
    if domain_name.lower() not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain: {domain_name}. Available: {list(DOMAIN_REGISTRY.keys())}")
    return DOMAIN_REGISTRY[domain_name.lower()]
