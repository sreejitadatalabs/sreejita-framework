"""
v2.0 domain bootstrap
Manual registration (safe, explicit)
"""

from sreejita.domains.registry import registry

from sreejita.domains import (
    retail,
    customer,
    finance,
    healthcare,
)

retail.register(registry)
customer.register(registry)
finance.register(registry)
healthcare.register(registry)
