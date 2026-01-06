# =====================================================
# DOMAIN ROUTER — UNIVERSAL (AUTHORITATIVE, FINAL)
# Sreejita Framework v3.6
# =====================================================

from typing import List, Dict, Any
import logging

# -----------------------------------------------------
# DOMAIN IMPORTS
# -----------------------------------------------------

from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector
from sreejita.domains.ecommerce import EcommerceDomain, EcommerceDomainDetector
from sreejita.domains.healthcare import HealthcareDomain, HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomain, MarketingDomainDetector
from sreejita.domains.hr import HRDomain, HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomain, SupplyChainDomainDetector

# 🚑 GENERIC FALLBACK (ABSOLUTE LAST RESORT)
from sreejita.domains.generic import GenericDomain, GenericDomainDetector

# -----------------------------------------------------
# CORE FRAMEWORK
# -----------------------------------------------------

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.router")

# =====================================================
# DOMAIN DETECTORS (ORDER MATTERS, GENERIC LAST)
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
    GenericDomainDetector(),  # 🚑 ABSOLUTE LAST
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
    "generic": GenericDomain(),
}

# =====================================================
# OBSERVABILITY
# =====================================================

_OBSERVERS: List[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    """
    Register a non-blocking observer for domain-decision events.
    Observers must NEVER raise.
    """
    if observer:
        _OBSERVERS.append(observer)


# =====================================================
# DOMAIN DECISION ENGINE (AUTHORITATIVE)
# =====================================================

def decide_domain(df) -> DecisionExplanation:
    """
    Determine the most appropriate domain for a dataset.

    GUARANTEES:
    - Detector confidence is authoritative
    - No re-scoring / no re-weighting
    - Generic is true last resort
    - Always returns DecisionExplanation
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # PHASE 1: RULE-BASED DETECTION (AUTHORITATIVE)
    # -------------------------------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)

            if not result or not result.domain:
                continue

            # Keep the BEST confidence per domain
            prev = rule_results.get(result.domain)
            if prev is None or result.confidence > prev["confidence"]:
                rule_results[result.domain] = {
                    "confidence": float(result.confidence or 0.0),
                    "signals": result.signals or {},
                    "detector": detector.__class__.__name__,
                }

        except Exception as e:
            log.debug(
                f"Detector {detector.__class__.__name__} failed: {e}"
            )

    # -------------------------------------------------
    # PHASE 2: AUTHORITATIVE SELECTION
    # -------------------------------------------------
    if rule_results:
        selected_domain, best = max(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
        )
        confidence = best["confidence"]
        meta = best
    else:
        selected_domain = "generic"
        confidence = 0.25
        meta = {"signals": {"fallback": True}}

    # -------------------------------------------------
    # HARD SAFETY CHECK
    # -------------------------------------------------
    if (
        selected_domain not in DOMAIN_IMPLEMENTATIONS
        or not isinstance(confidence, (int, float))
        or confidence < 0.25
    ):
        log.warning(
            "No confident domain detected — falling back to GENERIC domain"
        )
        selected_domain = "generic"
        confidence = 0.25
        meta = {"signals": {"fallback": True}}

    # -------------------------------------------------
    # BUILD ALTERNATIVES (EXPLAINABILITY)
    # -------------------------------------------------
    alternatives = [
        {
            "domain": d,
            "confidence": round(info["confidence"], 2),
        }
        for d, info in sorted(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )
        if d != selected_domain
    ]

    # -------------------------------------------------
    # DECISION OBJECT (STRICT CONTRACT)
    # -------------------------------------------------
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=round(confidence, 2),
        alternatives=alternatives,
        signals=meta.get("signals", {}),
        rules_applied=[
            "authoritative_detector_selection",
            "highest_confidence_wins",
            "generic_fallback_only_if_required",
        ],
        domain_scores={
            k: {"confidence": v["confidence"]}
            for k, v in rule_results.items()
        },
    )

    # -------------------------------------------------
    # ATTACH ENGINE (NEVER NULL)
    # -------------------------------------------------
    decision.engine = DOMAIN_IMPLEMENTATIONS[selected_domain]

    # -------------------------------------------------
    # DATASET FINGERPRINT (TRACEABILITY)
    # -------------------------------------------------
    decision.fingerprint = dataframe_fingerprint(df)

    # -------------------------------------------------
    # OBSERVABILITY HOOKS (NON-BLOCKING)
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
    if not domain:
        return df

    try:
        return domain.preprocess(df)
    except Exception:
        return df
