"""
Sreejita Framework – Domains Package (v2.x FINAL)

This module:
- Exposes public domain contracts (BaseDomain, detectors)
- Registers all domain implementations
- Provides backward-compatible shims for v1.x API
"""

# =========================
# Core domain contracts
# =========================

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import (
    BaseDomainDetector,
    DomainDetectionResult,
)

# =========================
# Registry (v2.x core)
# =========================

from sreejita.domains.registry import registry

# =========================
# Domain implementations
# =========================

from sreejita.domains.retail import RetailDomain
from sreejita.domains.customer import CustomerDomain
from sreejita.domains.ecommerce import EcommerceDomain
from sreejita.domains.finance import FinanceDomain
from sreejita.domains.ops import OpsDomain
from sreejita.domains.healthcare import HealthcareDomain
from sreejita.domains.marketing import MarketingDomain
from sreejita.domains.text import TextDomain  # REQUIRED for tests / fallback

# =========================
# Register domains (ONCE)
# =========================

registry.register("retail", RetailDomain)
registry.register("customer", CustomerDomain)
registry.register("ecommerce", EcommerceDomain)
registry.register("finance", FinanceDomain)
registry.register("ops", OpsDomain)
registry.register("healthcare", HealthcareDomain)
registry.register("marketing", MarketingDomain)
registry.register("text", TextDomain)

# =========================
# 🔁 BACKWARD COMPATIBILITY (v1.x)
# =========================

def get_domain(domain_name: str):
    """
    v1.x compatibility wrapper.
    Returns a NEW domain instance.
    """
    return registry.get_domain(domain_name)


# Read-only compatibility alias
DOMAIN_REGISTRY = {
    name: cls
    for name, cls in {
        "retail": RetailDomain,
        "customer": CustomerDomain,
        "ecommerce": EcommerceDomain,
        "finance": FinanceDomain,
        "ops": OpsDomain,
        "healthcare": HealthcareDomain,
        "marketing": MarketingDomain,
        "text": TextDomain,
    }.items()
}

# =========================
# Public exports
# =========================

__all__ = [
    # contracts
    "BaseDomain",
    "BaseDomainDetector",
    "DomainDetectionResult",

    # domains
    "RetailDomain",
    "CustomerDomain",
    "EcommerceDomain",
    "FinanceDomain",
    "OpsDomain",
    "HealthcareDomain",
    "MarketingDomain",
    "TextDomain",

    # v1.x compatibility
    "get_domain",
    "DOMAIN_REGISTRY",
]
