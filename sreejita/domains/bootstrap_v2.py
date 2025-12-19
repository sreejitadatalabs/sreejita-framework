"""
v2.x domain bootstrap
Explicit, deterministic registration
"""

from sreejita.domains.registry import registry

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

retail.register(registry)
ecommerce.register(registry)
customer.register(registry)
finance.register(registry)
healthcare.register(registry)
hr.register(registry)
supply_chain.register(registry)
marketing.register(registry)
