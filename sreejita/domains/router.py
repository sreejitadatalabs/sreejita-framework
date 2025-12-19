from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector
from sreejita.domains.healthcare import HealthcareDomain, HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomain, MarketingDomainDetector
from sreejita.domains.hr import HRDomain, HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomain, SupplyChainDomainDetector

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

# Phase-2 intelligence
from sreejita.domains.intelligence.detector_v2 import (
    compute_domain_scores,
    select_best_domain,
)

# =====================================================
# DOMAIN DETECTORS — PHASE 1 (CRITICAL)
# =====================================================

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    CustomerDomainDetector(),
    FinanceDomainDetector(),
    HealthcareDomainDetector(),
    MarketingDomainDetector(),
    HRDomainDetector(),              # ✅ FIX
    SupplyChainDomainDetector(),     # ✅ FIX
]

# =====================================================
# DOMAIN IMPLEMENTATIONS
# =====================================================

DOMAIN_IMPLEMENTATIONS = {
    "retail": RetailDomain(),
    "customer": CustomerDomain(),
    "finance": FinanceDomain(),
    "healthcare": HealthcareDomain(),
    "marketing": MarketingDomain(),
    "hr": HRDomain(),                # ✅ FIX
    "supply_chain": SupplyChainDomain(),  # ✅ FIX
}

# =====================================================
# OBSERVABILITY
# =====================================================

_OBSERVERS: list[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    _OBSERVERS.append(observer)

# =====================================================
# DOMAIN DECISION (v2.x)
# =====================================================

def decide_domain(df) -> DecisionExplanation:
    rule_results = {}

    # 1️⃣ Phase-1 rule-based detectors
    for detector in DOMAIN_DETECTORS:
        result = detector.detect(df)
        rule_results[result.domain] = {
            "confidence": result.confidence,
            "signals": result.signals,
        }

    # 2️⃣ Phase-2 intent-weighted scoring
    domain_scores = compute_domain_scores(df, rule_results)

    # 3️⃣ Select best domain
    selected_domain, confidence, meta = select_best_domain(domain_scores)

    # 4️⃣ Alternatives
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

    # 5️⃣ Build decision object
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


# =====================================================
# DOMAIN APPLICATION
# =====================================================

def apply_domain(df, domain_name: str):
    domain = DOMAIN_IMPLEMENTATIONS.get(domain_name)
    if domain:
        return domain.preprocess(df)
    return df
