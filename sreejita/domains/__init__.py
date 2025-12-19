"""
Sreejita Framework - Domain Modules

Only real, registered business domains are exposed here.
Capabilities (NLP, ML, etc.) are NOT domains.
"""

from .base import BaseDomain

from .retail import RetailDomain
from .ecommerce import EcommerceDomain
from .customer import CustomerDomain
from .finance import FinanceDomain
from .healthcare import HealthcareDomain
from .hr import HRDomain
from .supply_chain import SupplyChainDomain
from .marketing import MarketingDomain

__all__ = [
    "BaseDomain",
    "RetailDomain",
    "EcommerceDomain",
    "CustomerDomain",
    "FinanceDomain",
    "HealthcareDomain",
    "HRDomain",
    "SupplyChainDomain",
    "MarketingDomain",
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
