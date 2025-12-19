"""
Sreejita Framework - Domain Modules (v2.x)

This package exposes all domain engines for:
- auto-registration
- routing
- dynamic resolution
- external imports

IMPORTANT:
This file MUST stay in sync with registry + router.
"""

from .base import BaseDomain

# Core business domains
from .retail import RetailDomain
from .customer import CustomerDomain
from .finance import FinanceDomain
from .healthcare import HealthcareDomain
from .marketing import MarketingDomain

# Operations & workforce
from .supply_chain import SupplyChainDomain
from .hr import HRDomain

# Optional / legacy
from .ecommerce import EcommerceDomain
from .text import TextDomain

__all__ = [
    # Base
    "BaseDomain",

    # Core
    "RetailDomain",
    "CustomerDomain",
    "FinanceDomain",
    "HealthcareDomain",
    "MarketingDomain",

    # Ops / HR
    "SupplyChainDomain",
    "HRDomain",

    # Optional
    "EcommerceDomain",
    "TextDomain",
]

# -----------------------------------------------------
# v2.x DOMAIN REGISTRY (CLASS MAP ONLY)
# -----------------------------------------------------

DOMAIN_REGISTRY = {
    "retail": RetailDomain,
    "customer": CustomerDomain,
    "finance": FinanceDomain,
    "healthcare": HealthcareDomain,
    "marketing": MarketingDomain,
    "supply_chain": SupplyChainDomain,
    "hr": HRDomain,

    # optional / legacy
    "ecommerce": EcommerceDomain,
    "text": TextDomain,
}


def get_domain(domain_name: str):
    """
    Dynamically resolve a domain class by name.

    NOTE:
    Actual execution uses registry + router.
    This is a safe utility for external callers.
    """
    key = domain_name.lower()
    if key not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Available: {list(DOMAIN_REGISTRY.keys())}"
        )
    return DOMAIN_REGISTRY[key]
