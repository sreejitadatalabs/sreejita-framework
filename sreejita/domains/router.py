from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

from sreejita.domains.intelligence.detector_v2 import (
    compute_domain_scores,
    select_best_domain,
)

from sreejita.domains.retail import RetailDomainDetector
from sreejita.domains.customer import CustomerDomainDetector
from sreejita.domains.finance import FinanceDomainDetector
from sreejita.domains.ops import OpsDomainDetector
from sreejita.domains.healthcare import HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomainDetector

# ------------------------
# Domain detectors ONLY
# ------------------------

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    CustomerDomainDetector(),
    FinanceDomainDetector(),
    OpsDomainDetector(),
    HealthcareDomainDetector(),
    MarketingDomainDetector(),
]

_OBSERVERS: list[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    _OBSERVERS.append(observer)


def decide_domain(df) -> DecisionExplanation:
    """
    v2.x FINAL
    Domain detection ONLY.
    No domain instantiation here.
    """

    rule_results = {}

    # Phase 1: rule-based detectors
    for detector in DOMAIN_DETECTORS:
        result = detector.detect(df)
        rule_results[result.domain] = {
            "confidence": result.confidence,
            "signals": result.signals,
        }

    # Phase 2: intent-weighted scoring
    domain_scores = compute_domain_scores(df, rule_results)

    # Select best domain
    selected_domain, confidence, meta = select_best_domain(domain_scores)

    alternatives = [
        {"domain": d, "confidence": info["confidence"]}
        for d, info in sorted(
            domain_scores.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )
        if d != selected_domain
    ]

    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=confidence,
        alternatives=alternatives,
        signals=meta.get("signals", {}) if meta else {},
        rules_applied=[
            "rule_based_domain_detection",
            "intent_weighted_scoring",
            "highest_confidence_selection",
        ],
        domain_scores=domain_scores,
    )

    decision.fingerprint = dataframe_fingerprint(df)

    for observer in _OBSERVERS:
        observer.record(decision)

    return decision
