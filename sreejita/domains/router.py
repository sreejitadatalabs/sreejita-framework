from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver

# ------------------------
# Domain detectors
# ------------------------

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

# ------------------------
# Observability (Step 3.2)
# ------------------------

_OBSERVERS: list[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    """
    Register a decision observer (console, file, etc.)
    """
    _OBSERVERS.append(observer)


# ------------------------
# Domain decision (v2.3)
# ------------------------

def decide_domain(df) -> DecisionExplanation:
    results = []

    for detector in DOMAIN_DETECTORS:
        result = detector.detect(df)
        results.append(result)

    # Sort by confidence
    results.sort(key=lambda r: r.confidence, reverse=True)

    # No valid domain detected
    if not results or results[0].confidence <= 0:
        decision = DecisionExplanation(
            decision_type="domain_detection",
            selected_domain="unknown",
            confidence=0.0,
            alternatives=[],
            signals={},
            rules_applied=["no_domain_above_threshold"]
        )
    else:
        primary = results[0]

        decision = DecisionExplanation(
            decision_type="domain_detection",
            selected_domain=primary.domain,
            confidence=primary.confidence,
            alternatives=[
                {"domain": r.domain, "confidence": r.confidence}
                for r in results[1:]
            ],
            signals=primary.signals,
            rules_applied=[
                "rule_based_domain_detection",
                "highest_confidence_selection"
            ]
        )

    # ------------------------
    # OBSERVABILITY HOOK
    # ------------------------
    for observer in _OBSERVERS:
        observer.record(decision)

    return decision


# ------------------------
# Domain application
# ------------------------

def apply_domain(df, domain_name: str):
    domain = DOMAIN_IMPLEMENTATIONS.get(domain_name)
    if domain:
        return domain.preprocess(df)
    return df
