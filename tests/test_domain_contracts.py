"""
CI guard tests for v1.x public domain contracts.
These tests ensure backward compatibility.
"""

def test_domain_detectors_exist():
    # Retail
    from sreejita.domains.retail import RetailDomainDetector

    # Customer
    from sreejita.domains.customer import CustomerDomainDetector

    # Finance
    from sreejita.domains.finance import FinanceDomainDetector

    # Healthcare (if present)
    try:
        from sreejita.domains.healthcare import HealthcareDomainDetector
    except ImportError:
        pass  # optional domain


def test_domain_classes_exist():
    from sreejita.domains.retail import RetailDomain
    from sreejita.domains.customer import CustomerDomain
    from sreejita.domains.finance import FinanceDomain
