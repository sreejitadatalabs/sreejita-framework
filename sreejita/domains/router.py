from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector
from sreejita.domains.ecommerce import EcommerceDomain, EcommerceDomainDetector
from sreejita.domains.healthcare import HealthcareDomain, HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomain, MarketingDomainDetector
from sreejita.domains.hr import HRDomain, HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomain, SupplyChainDomainDetector

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

from sreejita.domains.intelligence.detector_v2 import (
    compute_domain_scores,
    select_best_domain,
)

# =====================================================
# DOMAIN DETECTORS (RULE-BASED)
# =====================================================
# NOTE:
# Domains are explicitly registered here for clarity and auditability.
# Dynamic discovery is intentionally avoided in v3.6 for determinism.

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    CustomerDomainDetector(),
    FinanceDomainDetector(),
    EcommerceDomainDetector(),
    HealthcareDomainDetector(),
    MarketingDomainDetector(),
    HRDomainDetector(),
    SupplyChainDomainDetector(),
]

# =====================================================
# DOMAIN IMPLEMENTATIONS
# =====================================================

DOMAIN_IMPLEMENTATIONS = {
    "retail": RetailDomain(),
    "customer": CustomerDomain(),
    "finance": FinanceDomain(),
    "ecommerce": EcommerceDomain(),
    "healthcare": HealthcareDomain(),
    "marketing": MarketingDomain(),
    "hr": HRDomain(),
    "supply_chain": SupplyChainDomain(),
}

# =====================================================
# OBSERVABILITY
# =====================================================

_OBSERVERS: list[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    """
    Register an observer for domain-decision events.
    Observers must be non-blocking and side-effect safe.
    """
    _OBSERVERS.append(observer)


# =====================================================
# DOMAIN DECISION ENGINE
# =====================================================

def decide_domain(df) -> DecisionExplanation:
    """
    Determine the most appropriate domain for a dataset.

    This function:
    1. Runs rule-based detectors
    2. Computes weighted domain scores
    3. Selects the highest-confidence domain
    4. Produces a fully explainable decision object
    """

    rule_results = {}

    # ------------------------
    # Phase 1: Rule Detection
    # ------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)
            if not result or not result.domain:
                continue

            rule_results[result.domain] = {
                "confidence": result.confidence,
                "signals": result.signals,
            }
        except Exception:
            # Detector failure should not break domain resolution
            continue

    # ------------------------
    # Phase 2: Score & Select
    # ------------------------
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
        signals=meta.get("signals", {}) if meta else {},
        rules_applied=[
            "rule_based_domain_detection",
            "intent_weighted_scoring",
            "highest_confidence_selection",
        ],
        domain_scores=domain_scores,
    )

    # Attach executable domain engine (critical for orchestration)
    decision.engine = DOMAIN_IMPLEMENTATIONS.get(selected_domain)

    # Attach dataset fingerprint for traceability
    decision.fingerprint = dataframe_fingerprint(df)

    # ------------------------
    # Observability (Non-Blocking)
    # ------------------------
    for observer in _OBSERVERS:
        try:
            observer.record(decision)
        except Exception:
            # Observers must never break core execution
            pass

    return decision


# =====================================================
# DOMAIN PREPROCESSING
# =====================================================

def apply_domain(df, domain_name: str):
    """
    Apply domain-specific preprocessing ONLY.

    This does NOT:
    - calculate KPIs
    - generate insights
    - run recommendations

    It exists as a convenience hook for pipelines.
    """
    domain = DOMAIN_IMPLEMENTATIONS.get(domain_name)
    if domain:
        return domain.preprocess(df)
    return df
