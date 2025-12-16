from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    CustomerDomainDetector(),
    FinanceDomainDetector(),
]

DOMAIN_IMPLEMENTATIONS = {
    "retail": RetailDomain(),
    "customer": CustomerDomain(),
    "finance": FinanceDomain(),
}
