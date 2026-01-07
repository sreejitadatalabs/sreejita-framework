"""
Bootstrap v2 — Domain Registration (Authoritative)
Sreejita Framework v3.6

Purpose:
- Deterministic domain registration
- Explicit imports (NO dynamic discovery)
- Safe for CLI, UI, batch, scheduler
- Idempotent (can be imported many times)
"""

from typing import Iterable
import logging

from sreejita.domains.registry import registry

log = logging.getLogger("sreejita.bootstrap")


# =====================================================
# DOMAIN MODULE IMPORTS (EXPLICIT & ORDERED)
# =====================================================
# ⚠️ Order matters only for logging readability
# Registry itself is order-agnostic

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


# =====================================================
# SAFE REGISTRATION HELPER
# =====================================================
def _safe_register(domain_module, registry):
    """
    Register a domain module safely.

    Guarantees:
    - No duplicate domain overwrite
    - Clear structured logging
    - NEVER crashes runtime
    """

    module_name = getattr(domain_module, "__name__", str(domain_module))

    try:
        if not hasattr(domain_module, "register"):
            raise AttributeError(
                f"{module_name} does not expose register(registry)"
            )

        # Let registry itself guard duplicates
        domain_module.register(registry)

        log.info("✅ Domain registered: %s", module_name)

    except RuntimeError as e:
        # Expected case: already registered
        log.debug(
            "ℹ️ Domain already registered: %s (%s)",
            module_name,
            str(e),
        )

    except Exception as e:
        # Hard failure — log loudly, continue safely
        log.error(
            "❌ Domain registration failed: %s | %s",
            module_name,
            str(e),
            exc_info=True,
        )


# =====================================================
# BOOTSTRAP ENTRYPOINT (IDEMPOTENT)
# =====================================================
def bootstrap_domains():
    """
    Bootstrap all domains.

    SAFE:
    - Can be called multiple times
    - Registry guarantees no duplicates
    - Never raises
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


# =====================================================
# AUTO-BOOTSTRAP (CRITICAL)
# =====================================================
# Ensures:
# - CLI works
# - UI works
# - Batch works
# - Scheduler works
# - Importing multiple times is safe
bootstrap_domains()
