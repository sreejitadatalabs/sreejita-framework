from enum import Enum
from dataclasses import dataclass
from typing import Dict, Set


# =====================================================
# UNIVERSAL CAPABILITY ENUM
# =====================================================

class Capability(str, Enum):
    """
    Core pillars of Universal Domain Intelligence.

    Every KPI, Insight, and Recommendation in Sreejita
    MUST map to exactly one Capability.
    """

    # 1. SCALE & DEMAND
    VOLUME = "volume"              # Counts, Throughput, Load, Patients, Transactions

    # 2. EFFICIENCY & FLOW
    TIME_FLOW = "time_flow"        # LOS, TAT, Wait Time, Cycle Time, Lead Time

    # 3. ECONOMICS
    COST = "cost"                  # Unit cost, Spend, Revenue, Margin

    # 4. QUALITY & OUTCOME
    QUALITY = "quality"            # Errors, Readmissions, Defects, Returns

    # 5. EXECUTION DISCIPLINE
    VARIANCE = "variance"          # Consistency across entities or time

    # 6. AVAILABILITY & REACH
    ACCESS = "access"              # Utilization, Coverage, Availability

    # 7. GOVERNANCE FOUNDATION
    DATA_TRUST = "data_trust"      # Completeness, Validity, Sample size


# =====================================================
# CAPABILITY SPECIFICATION (AUTHORITATIVE)
# =====================================================

@dataclass(frozen=True)
class CapabilitySpec:
    """
    Canonical metadata describing how a capability
    behaves across all domains.
    """
    label: str
    executive_intent: str
    min_confidence: float
    supports_trend: bool
    supports_benchmark: bool
    aggregation_safe: bool


CAPABILITY_SPECS: Dict[Capability, CapabilitySpec] = {

    Capability.DATA_TRUST: CapabilitySpec(
        label="Data Trust & Reliability",
        executive_intent="Governance confidence and decision safety",
        min_confidence=0.85,
        supports_trend=False,
        supports_benchmark=False,
        aggregation_safe=False,
    ),

    Capability.QUALITY: CapabilitySpec(
        label="Quality & Outcome",
        executive_intent="Risk, safety, and outcome integrity",
        min_confidence=0.75,
        supports_trend=True,
        supports_benchmark=True,
        aggregation_safe=True,
    ),

    Capability.COST: CapabilitySpec(
        label="Cost & Financial Impact",
        executive_intent="Economic efficiency and margin control",
        min_confidence=0.70,
        supports_trend=True,
        supports_benchmark=True,
        aggregation_safe=True,
    ),

    Capability.TIME_FLOW: CapabilitySpec(
        label="Time & Flow Efficiency",
        executive_intent="Operational throughput and efficiency",
        min_confidence=0.70,
        supports_trend=True,
        supports_benchmark=True,
        aggregation_safe=True,
    ),

    Capability.VOLUME: CapabilitySpec(
        label="Scale & Demand",
        executive_intent="Operational scale and exposure",
        min_confidence=0.60,
        supports_trend=True,
        supports_benchmark=False,
        aggregation_safe=True,
    ),

    Capability.VARIANCE: CapabilitySpec(
        label="Execution Consistency",
        executive_intent="Standardization and control",
        min_confidence=0.75,
        supports_trend=False,
        supports_benchmark=True,
        aggregation_safe=False,
    ),

    Capability.ACCESS: CapabilitySpec(
        label="Access & Availability",
        executive_intent="Reach, utilization, and service availability",
        min_confidence=0.65,
        supports_trend=True,
        supports_benchmark=False,
        aggregation_safe=True,
    ),
}


# =====================================================
# EXECUTIVE PRIORITY MODEL
# =====================================================

CAPABILITY_PRIORITY: Dict[Capability, int] = {
    Capability.DATA_TRUST: 1,   # Foundation
    Capability.QUALITY: 2,      # Risk & Safety
    Capability.COST: 3,         # Financial Impact
    Capability.TIME_FLOW: 4,    # Efficiency
    Capability.VOLUME: 5,       # Context
    Capability.VARIANCE: 6,     # Governance
    Capability.ACCESS: 7,       # Growth & Reach
}


RISK_CAPABILITIES: Set[Capability] = {
    Capability.DATA_TRUST,
    Capability.QUALITY,
    Capability.VARIANCE,
}

PERFORMANCE_CAPABILITIES: Set[Capability] = {
    Capability.VOLUME,
    Capability.TIME_FLOW,
    Capability.COST,
    Capability.ACCESS,
}


# =====================================================
# SAFE ACCESS HELPERS (CANONICAL API)
# =====================================================

def get_capability_spec(cap: Capability) -> CapabilitySpec:
    """Returns authoritative metadata for a capability."""
    return CAPABILITY_SPECS[cap]


def is_risk_capability(cap: Capability) -> bool:
    return cap in RISK_CAPABILITIES


def is_performance_capability(cap: Capability) -> bool:
    return cap in PERFORMANCE_CAPABILITIES


def min_confidence_required(cap: Capability) -> float:
    return CAPABILITY_SPECS[cap].min_confidence


def supports_trend(cap: Capability) -> bool:
    return CAPABILITY_SPECS[cap].supports_trend


def supports_benchmark(cap: Capability) -> bool:
    return CAPABILITY_SPECS[cap].supports_benchmark


def is_aggregation_safe(cap: Capability) -> bool:
    return CAPABILITY_SPECS[cap].aggregation_safe


def capability_priority(cap: Capability) -> int:
    """Lower number = higher executive importance."""
    return CAPABILITY_PRIORITY.get(cap, 99)
