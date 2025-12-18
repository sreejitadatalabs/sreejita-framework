from sreejita.domains.registry import registry

from sreejita.domains.retail import RetailDomain
from sreejita.domains.customer import CustomerDomain
from sreejita.domains.finance import FinanceDomain
from sreejita.domains.ops import OpsDomain
from sreejita.domains.healthcare import HealthcareDomain
from sreejita.domains.marketing import MarketingDomain

registry.register("retail", RetailDomain)
registry.register("customer", CustomerDomain)
registry.register("finance", FinanceDomain)
registry.register("ops", OpsDomain)
registry.register("healthcare", HealthcareDomain)
registry.register("marketing", MarketingDomain)
