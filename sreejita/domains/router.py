# =====================================================
# DOMAIN ROUTER — UNIVERSAL (FINAL, ENFORCED)
# Sreejita Framework v3.6
# =====================================================

from typing import List, Dict, Any
import logging

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

log = logging.getLogger("sreejita.router")

# =====================================================
# DOMAIN DETECTORS (RULE-BASED, DETERMINISTIC)
# =====================================================

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
# DOMAIN IMPLEMENTATIONS (SINGLETONS)
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

_OBSERVERS: List[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    """
    Register a non-blocking observer for domain-decision events.
    """
    _OBSERVERS.append(observer)


# =====================================================
# DOMAIN DECISION ENGINE (AUTHORITATIVE)
# =====================================================

def decide_domain(df) -> DecisionExplanation:
    """
    Determine the most appropriate domain for a dataset.

    GUARANTEES:
    - Always returns a DecisionExplanation
    - Always attaches a valid domain engine
    - Never raises for unknown datasets
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # Phase 1: Rule-Based Detection
    # -------------------------------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)
            if not result or not result.domain:
                continue

            rule_results[result.domain] = {
                "confidence": float(result.confidence or 0.0),
                "signals": result.signals or {},
            }

        except Exception as e:
            log.debug(f"Detector {detector.__class__.__name__} failed: {e}")
            continue

    # -------------------------------------------------
    # Phase 2: Weighted Scoring & Selection
    # -------------------------------------------------
    domain_scores = compute_domain_scores(df, rule_results)

    selected_domain, confidence, meta = select_best_domain(domain_scores)

    # -------------------------------------------------
    # 🚑 HARD FALLBACK (CRITICAL)
    # -------------------------------------------------
    if not selected_domain or selected_domain not in DOMAIN_IMPLEMENTATIONS:
        log.warning(
            "No strong domain detected — falling back to healthcare (generic-safe)"
        )
        selected_domain = "healthcare"
        confidence = round(
            max(
                rule_results.get("healthcare", {}).get("confidence", 0.3),
                0.3,
            ),
            2,
        )
        meta = meta or {"signals": {}}

    # -------------------------------------------------
    # Build Alternatives (Explainability)
    # -------------------------------------------------
    alternatives = [
        {
            "domain": d,
            "confidence": round(info.get("confidence", 0.0), 2),
        }
        for d, info in sorted(
            domain_scores.items(),
            key=lambda x: x[1].get("confidence", 0),
            reverse=True,
        )
        if d != selected_domain
    ]

    # -------------------------------------------------
    # Decision Object (AUTHORITATIVE CONTRACT)
    # -------------------------------------------------
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=round(confidence or 0.0, 2),
        alternatives=alternatives,
        signals=meta.get("signals", {}) if isinstance(meta, dict) else {},
        rules_applied=[
            "rule_based_domain_detection",
            "intent_weighted_scoring",
            "highest_confidence_selection",
            "safe_domain_fallback",
        ],
        domain_scores=domain_scores,
    )

    # -------------------------------------------------
    # Attach Engine (🚨 NEVER NULL)
    # -------------------------------------------------
    decision.engine = DOMAIN_IMPLEMENTATIONS[selected_domain]

    # -------------------------------------------------
    # Dataset Fingerprint (Traceability)
    # -------------------------------------------------
    decision.fingerprint = dataframe_fingerprint(df)

    # -------------------------------------------------
    # Observability Hooks (Non-Blocking)
    # -------------------------------------------------
    for observer in _OBSERVERS:
        try:
            observer.record(decision)
        except Exception:
            pass

    return decision


# =====================================================
# DOMAIN PREPROCESSING (OPTIONAL UTILITY)
# =====================================================

def apply_domain(df, domain_name: str):
    """
    Apply ONLY domain-specific preprocessing.

    This does NOT:
    - calculate KPIs
    - generate insights
    - generate recommendations
    """
    domain = DOMAIN_IMPLEMENTATIONS.get(domain_name)
    if domain:
        try:
            return domain.preprocess(df)
        except Exception:
            return df
    return df
