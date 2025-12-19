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
    marketing,
    hr,
    supply_chain,
)

retail.register(registry)
customer.register(registry)
finance.register(registry)
healthcare.register(registry)
hr.register(registry)
supply_chain.register(registry)
marketing.register(registry)


