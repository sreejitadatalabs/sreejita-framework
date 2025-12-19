"""
Auto-registration for v2.x domains.

Imports domain modules and triggers their register() hooks.
Safe, explicit, debuggable.
"""

from importlib import import_module
from typing import Iterable

from sreejita.domains.registry import registry


# List of domain modules to auto-register
DEFAULT_DOMAIN_MODULES = [
    "sreejita.domains.retail",
    "sreejita.domains.customer",
    "sreejita.domains.finance",
    "sreejita.domains.healthcare",
    "sreejita.domains.supply_chain",
    "sreejita.domains.hr",
]


def auto_register_domains(
    modules: Iterable[str] = DEFAULT_DOMAIN_MODULES,
) -> None:
    """
    Import domain modules and call their register(registry) hook.
    """

    for module_path in modules:
        try:
            module = import_module(module_path)
        except Exception:
            # Import failed → skip safely
            continue

        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            try:
                register_fn(registry)
            except Exception:
                # Registration must never crash startup
                continue
