"""
Sreejita Framework - Domains Package (v2.x FINAL)

Responsibilities:
1. Export core domain contracts (public API)
2. Register all concrete domain implementations
"""

# -------------------------
# PUBLIC DOMAIN CONTRACTS
# -------------------------

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import (
    BaseDomainDetector,
    DomainDetectionResult,
)

# -------------------------
# REGISTRY (SINGLETON)
# -------------------------

from sreejita.domains.registry import registry

# -------------------------
# CONCRETE DOMAINS
# -------------------------

from sreejita.domains.retail import RetailDomain
from sreejita.domains.customer import CustomerDomain
from sreejita.domains.finance import FinanceDomain
from sreejita.domains.ops import OpsDomain
from sreejita.domains.healthcare import HealthcareDomain
from sreejita.domains.marketing import MarketingDomain
from sreejita.domains.ecommerce import EcommerceDomain

# -------------------------
# REGISTER DOMAINS (ONCE)
# -------------------------

registry.register("retail", RetailDomain)
registry.register("customer", CustomerDomain)
registry.register("finance", FinanceDomain)
registry.register("ops", OpsDomain)
registry.register("healthcare", HealthcareDomain)
registry.register("marketing", MarketingDomain)
registry.register("ecommerce", EcommerceDomain)

# -------------------------
# PUBLIC EXPORTS
# -------------------------

__all__ = [
    # contracts
    "BaseDomain",
    "BaseDomainDetector",
    "DomainDetectionResult",

    # domains
    "RetailDomain",
    "CustomerDomain",
    "FinanceDomain",
    "OpsDomain",
    "HealthcareDomain",
    "MarketingDomain",
    "EcommerceDomain",
]
