from sreejita.core.decision import DecisionExplanation
from sreejita.core.fingerprint import dataframe_fingerprint

from sreejita.domains.retail import RetailDomainDetector
from sreejita.domains.ecommerce import EcommerceDomainDetector
from sreejita.domains.customer import CustomerDomainDetector
from sreejita.domains.finance import FinanceDomainDetector
from sreejita.domains.healthcare import HealthcareDomainDetector
from sreejita.domains.hr import HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomainDetector
from sreejita.domains.marketing import MarketingDomainDetector

from sreejita.domains.intelligence.detector_v2 import (
    compute_domain_scores,
    select_best_domain,
)

# ------------------------
# Phase-1 Rule Detectors
# ------------------------

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    EcommerceDomainDetector(),
    CustomerDomainDetector(),
    FinanceDomainDetector(),
    HealthcareDomainDetector(),
    HRDomainDetector(),
    SupplyChainDomainDetector(),
    MarketingDomainDetector(),
]


def decide_domain(df) -> DecisionExplanation:
    rule_results = {}

    # Phase 1 — rule-based detection
    for detector in DOMAIN_DETECTORS:
        result = detector.detect(df)
        rule_results[result.domain] = {
            "confidence": result.confidence,
            "signals": result.signals,
        }

    # Phase 2 — intent + dominance intelligence
    domain_scores = compute_domain_scores(df, rule_results)

    selected_domain, confidence, meta = select_best_domain(domain_scores)

    alternatives = [
        {
            "domain": d,
            "confidence": info["confidence"],
        }
        for d, info in sorted(
            domain_scores.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )
        if d != selected_domain
    ]

    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=confidence,
        alternatives=alternatives,
        signals=meta.get("signals", {}),
        rules_applied=[
            "rule_based_detection",
            "dominance_protected_intent_scoring",
        ],
        domain_scores=domain_scores,
    )

    decision.fingerprint = dataframe_fingerprint(df)

    return decision
