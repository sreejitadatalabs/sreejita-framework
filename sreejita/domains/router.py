# =====================================================
# DOMAIN ROUTER — UNIVERSAL (AUTHORITATIVE, FINAL)
# Sreejita Framework v3.6
# =====================================================

from typing import List, Dict, Any
import logging

import pandas as pd

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

# 🚑 GENERIC — ABSOLUTE LAST RESORT
from sreejita.domains.generic import GenericDomain, GenericDomainDetector

# -----------------------------------------------------
# CORE FRAMEWORK
# -----------------------------------------------------

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.router")

# =====================================================
# CONFIG (LOCKED)
# =====================================================

MIN_DOMAIN_CONFIDENCE = 0.45  # 🚨 hard guardrail

# =====================================================
# DOMAIN DETECTORS (GENERIC EXCLUDED FROM COMPETITION)
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
# DOMAIN IMPLEMENTATION FACTORY (DETERMINISTIC)
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
    try:
        return cls() if cls else GenericDomain()
    except Exception:
        return GenericDomain()

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

def decide_domain(df: pd.DataFrame) -> DecisionExplanation:
    """
    Determine the most appropriate domain for a dataset.

    GUARANTEES:
    - Generic NEVER competes
    - Healthcare cannot be overridden by Generic
    - Deterministic highest-confidence win
    - Minimum confidence enforced exactly once
    - Always returns DecisionExplanation
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # PHASE 1 — RULE-BASED DETECTION
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
    # PHASE 2 — SELECT BEST DOMAIN (NON-GENERIC)
    # -------------------------------------------------
    selected_domain: str = ""
    confidence: float = 0.0
    meta: Dict[str, Any] = {}

    if rule_results:
        selected_domain, meta = max(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
        )
        confidence = float(meta.get("confidence", 0.0))

    # -------------------------------------------------
    # PHASE 3 — HARD FALLBACK TO GENERIC (ONLY HERE)
    # -------------------------------------------------
    if (
        not selected_domain
        or confidence < MIN_DOMAIN_CONFIDENCE
        or selected_domain not in _DOMAIN_FACTORY
    ):
        generic = GENERIC_DETECTOR.detect(df)

        selected_domain = "generic"
        confidence = round(float(getattr(generic, "confidence", 0.25)), 2)
        meta = {
            "signals": getattr(generic, "signals", {"fallback": True}),
            "detector": "GenericDomainDetector",
        }

    # -------------------------------------------------
    # EXPLAINABILITY — ALTERNATIVES
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
            "rule_based_detection",
            "highest_confidence_wins",
            "minimum_confidence_gate",
            "generic_last_resort_only",
        ],
        domain_scores={
            d: {"confidence": v["confidence"]}
            for d, v in rule_results.items()
        },
    )

    # -------------------------------------------------
    # ENGINE ATTACHMENT (SAFE, LAZY)
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

def apply_domain(df: pd.DataFrame, domain_name: str):
    """
    Apply ONLY domain-level preprocessing.
    No KPIs, no insights, no visuals.
    """
    cls = _DOMAIN_FACTORY.get(domain_name)
    if not cls:
        return df

    try:
        return cls().preprocess(df)
    except Exception:
        return df
