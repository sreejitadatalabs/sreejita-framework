"""
Sreejita Framework - Domains Package (v2.x FINAL)

Exports public domain contracts AND registers domains.
"""

# -------------------------
# PUBLIC CONTRACTS
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
# DOMAIN IMPLEMENTATIONS
# -------------------------

from sreejita.domains.retail import RetailDomain
from sreejita.domains.customer import CustomerDomain
from sreejita.domains.finance import FinanceDomain
from sreejita.domains.ops import OpsDomain
from sreejita.domains.healthcare import HealthcareDomain
from sreejita.domains.marketing import MarketingDomain
from sreejita.domains.ecommerce import EcommerceDomain
from sreejita.domains.text import TextDomain   # 🔒 REQUIRED

# -------------------------
# REGISTER DOMAINS
# -------------------------

registry.register("retail", RetailDomain)
registry.register("customer", CustomerDomain)
registry.register("finance", FinanceDomain)
registry.register("ops", OpsDomain)
registry.register("healthcare", HealthcareDomain)
registry.register("marketing", MarketingDomain)
registry.register("ecommerce", EcommerceDomain)
registry.register("text", TextDomain)  # 🔒 REQUIRED

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
    "TextDomain",
]
