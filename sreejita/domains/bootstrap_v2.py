"""
Bootstrap v2 — Domain Registration (Authoritative)
Sreejita Framework v3.6
"""

from typing import Iterable
import logging

from sreejita.domains.registry import registry

log = logging.getLogger("sreejita.bootstrap")


# -------------------------------------------------
# DOMAIN MODULE IMPORTS (EXPLICIT & ORDERED)
# -------------------------------------------------

from sreejita.domains import (
    retail,
    ecommerce,
    customer,
    finance,
    healthcare,
    hr,
    supply_chain,
    marketing,
)


# -------------------------------------------------
# SAFE REGISTRATION HELPER
# -------------------------------------------------

def _safe_register(domain_module, registry):
    """
    Register a domain module safely.

    Guarantees:
    - No duplicate registration
    - Clear logging
    - Never crashes production
    """
    name = getattr(domain_module, "__name__", str(domain_module))

    try:
        if not hasattr(domain_module, "register"):
            raise AttributeError(f"{name} has no register() function")

        domain_module.register(registry)
        log.info(f"✅ Registered domain: {name}")

    except Exception as e:
        # Fail loudly in logs, but do not crash runtime
        log.error(f"❌ Failed to register domain {name}: {e}", exc_info=True)


# -------------------------------------------------
# BOOTSTRAP ENTRYPOINT (IDEMPOTENT)
# -------------------------------------------------

def bootstrap_domains():
    """
    Bootstrap all domain registrations.

    Safe to call multiple times.
    """

    domain_modules: Iterable = [
        retail,
        ecommerce,
        customer,
        finance,
        healthcare,
        hr,
        supply_chain,
        marketing,
    ]

    for module in domain_modules:
        _safe_register(module, registry)


# -------------------------------------------------
# AUTO-BOOTSTRAP (SAFE)
# -------------------------------------------------

# Ensures:
# - CLI works
# - UI works
# - Tests work
# - Multiple imports are safe
bootstrap_domains()
