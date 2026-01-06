# =====================================================
# DOMAIN ROUTER — UNIVERSAL (AUTHORITATIVE, FINAL)
# Sreejita Framework v3.6
# =====================================================

from typing import List, Dict, Any
import logging

# -----------------------------------------------------
# DOMAIN IMPORTS (DETECTORS ONLY)
# -----------------------------------------------------

from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector
from sreejita.domains.ecommerce import EcommerceDomain, EcommerceDomainDetector
from sreejita.domains.healthcare import HealthcareDomain, HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomain, MarketingDomainDetector
from sreejita.domains.hr import HRDomain, HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomain, SupplyChainDomainDetector

# 🚑 GENERIC (FALLBACK ONLY — NEVER COMPETES)
from sreejita.domains.generic import GenericDomain, GenericDomainDetector

# -----------------------------------------------------
# CORE FRAMEWORK
# -----------------------------------------------------

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.router")

# =====================================================
# CONFIG
# =====================================================

MIN_DOMAIN_CONFIDENCE = 0.45   # 🔒 critical guardrail

# =====================================================
# DOMAIN DETECTORS (GENERIC EXCLUDED)
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

GENERIC_DETECTOR = GenericDomainDetector()

# =====================================================
# DOMAIN IMPLEMENTATION FACTORY (LAZY)
# =====================================================

_DOMAIN_FACTORY = {
    "retail": RetailDomain,
    "customer": CustomerDomain,
    "finance": FinanceDomain,
    "ecommerce": EcommerceDomain,
    "healthcare": HealthcareDomain,
    "marketing": MarketingDomain,
    "hr": HRDomain,
    "supply_chain": SupplyChainDomain,
    "generic": GenericDomain,
}

def _get_domain_engine(name: str):
    cls = _DOMAIN_FACTORY.get(name)
    return cls() if cls else GenericDomain()

# =====================================================
# OBSERVABILITY
# =====================================================

_OBSERVERS: List[DecisionObserver] = []

def register_observer(observer: DecisionObserver):
    if observer:
        _OBSERVERS.append(observer)

# =====================================================
# DOMAIN DECISION ENGINE (AUTHORITATIVE)
# =====================================================

def decide_domain(df) -> DecisionExplanation:
    """
    Determine the most appropriate domain for a dataset.

    GUARANTEES:
    - Generic NEVER competes
    - Minimum confidence enforced
    - Deterministic selection
    - Always returns DecisionExplanation
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # PHASE 1: RULE-BASED DETECTION
    # -------------------------------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)

            if not result or not result.domain:
                continue

            prev = rule_results.get(result.domain)
            if prev is None or result.confidence > prev["confidence"]:
                rule_results[result.domain] = {
                    "confidence": float(result.confidence or 0.0),
                    "signals": result.signals or {},
                    "detector": detector.__class__.__name__,
                }

        except Exception as e:
            log.debug(f"{detector.__class__.__name__} failed: {e}")

    # -------------------------------------------------
    # PHASE 2: CONFIDENCE-GATED SELECTION
    # -------------------------------------------------
    selected_domain = None
    confidence = 0.0
    meta: Dict[str, Any] = {}

    if rule_results:
        selected_domain, best = max(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
        )
        confidence = best["confidence"]
        meta = best

    # -------------------------------------------------
    # PHASE 3: HARD FALLBACK TO GENERIC
    # -------------------------------------------------
    if (
        not selected_domain
        or confidence < MIN_DOMAIN_CONFIDENCE
        or selected_domain not in _DOMAIN_FACTORY
    ):
        generic = GENERIC_DETECTOR.detect(df)
        selected_domain = "generic"
        confidence = round(float(generic.confidence or 0.25), 2)
        meta = {
            "signals": getattr(generic, "signals", {"fallback": True}),
            "detector": "GenericDomainDetector",
        }

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
    # DECISION OBJECT
    # -------------------------------------------------
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=round(confidence, 2),
        alternatives=alternatives,
        signals=meta.get("signals", {}),
        rules_applied=[
            "rule_based_detection",
            "minimum_confidence_gate",
            "generic_fallback_only",
        ],
        domain_scores={
            k: {"confidence": v["confidence"]}
            for k, v in rule_results.items()
        },
    )

    # -------------------------------------------------
    # ATTACH ENGINE (LAZY, SAFE)
    # -------------------------------------------------
    decision.engine = _get_domain_engine(selected_domain)

    # -------------------------------------------------
    # TRACEABILITY
    # -------------------------------------------------
    decision.fingerprint = dataframe_fingerprint(df)

    # -------------------------------------------------
    # OBSERVABILITY (NON-BLOCKING)
    # -------------------------------------------------
    for observer in _OBSERVERS:
        try:
            observer.record(decision)
        except Exception:
            pass

    return decision

# =====================================================
# DOMAIN PREPROCESSING (UTILITY)
# =====================================================

def apply_domain(df, domain_name: str):
    domain = _DOMAIN_FACTORY.get(domain_name)
    if not domain:
        return df

    try:
        return domain().preprocess(df)
    except Exception:
        return df
