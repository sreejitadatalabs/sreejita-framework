"""
Sreejita Framework - Domain Initialization (v2.x FINAL)

This file ensures all domain modules are imported
and registered into the SINGLE registry instance.
"""

from sreejita.domains.registry import registry

from sreejita.domains.retail import RetailDomain
from sreejita.domains.ecommerce import EcommerceDomain
from sreejita.domains.customer import CustomerDomain
from sreejita.domains.marketing import MarketingDomain
from sreejita.domains.healthcare import HealthcareDomain
from sreejita.domains.ops import OpsDomain
from sreejita.domains.finance import FinanceDomain

# 🔒 REGISTER DOMAINS (ONCE)
registry.register("retail", RetailDomain)
registry.register("ecommerce", EcommerceDomain)
registry.register("customer", CustomerDomain)
registry.register("marketing", MarketingDomain)
registry.register("healthcare", HealthcareDomain)
registry.register("ops", OpsDomain)
registry.register("finance", FinanceDomain)

__all__ = [
    "RetailDomain",
    "EcommerceDomain",
    "CustomerDomain",
    "MarketingDomain",
    "HealthcareDomain",
    "OpsDomain",
    "FinanceDomain",
]
