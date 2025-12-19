"""
Sreejita Framework - Domain Modules
"""

from .retail import RetailDomain
from .customer import CustomerDomain
from .finance import FinanceDomain
from .healthcare import HealthcareDomain
from .ecommerce import EcommerceDomain
from .marketing import MarketingDomain
from .hr import HRDomain
from .supply_chain import SupplyChainDomain
from .base import BaseDomain

__all__ = [
    "RetailDomain",
    "CustomerDomain",
    "FinanceDomain",
    "HealthcareDomain",
    "EcommerceDomain",
    "MarketingDomain",
    "HRDomain",
    "SupplyChainDomain",
    "BaseDomain",
]

DOMAIN_REGISTRY = {
    "retail": RetailDomain,
    "customer": CustomerDomain,
    "finance": FinanceDomain,
    "ecommerce": EcommerceDomain,
    "healthcare": HealthcareDomain,
    "marketing": MarketingDomain,
    "hr": HRDomain,                   # ✅ FIX
    "supply_chain": SupplyChainDomain # ✅ FIX
}


def get_domain(domain_name):
    if domain_name.lower() not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Available: {list(DOMAIN_REGISTRY.keys())}"
        )
    return DOMAIN_REGISTRY[domain_name.lower()]
