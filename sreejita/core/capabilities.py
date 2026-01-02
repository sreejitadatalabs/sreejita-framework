from enum import Enum
from typing import Dict, List, Set

# =====================================================
# UNIVERSAL CAPABILITY ENUM
# =====================================================

class Capability(str, Enum):
    """
    The Core Pillars of Universal Domain Intelligence.
    Every KPI in Sreejita must map to one of these capabilities
    to ensure the Executive Cognition layer can process it.
    """
    
    # 1. SCALE & DEMAND
    VOLUME = "volume"              # Counts, Throughput, Load, Patients, Transactions
    
    # 2. EFFICIENCY & FLOW
    TIME_FLOW = "time_flow"        # Duration, LOS, TAT, Wait Time, Lead Time
    
    # 3. ECONOMICS
    COST = "cost"                  # Unit cost, Total spend, Margin, Revenue, Billing
    
    # 4. QUALITY & OUTCOME
    QUALITY = "quality"            # Errors, Readmissions, Returns, Success Rate
    
    # 5. EXECUTION DISCIPLINE
    VARIANCE = "variance"          # Consistency, Entity performance, Standardization
    
    # 6. AVAILABILITY
    ACCESS = "access"              # Utilization, Reach, Appointment availability
    
    # 7. GOVERNANCE FOUNDATION
    DATA_TRUST = "data_trust"      # Completeness, Validity, Statistical significance


# =====================================================
# CAPABILITY RELATIONSHIPS (META-LOGIC)
# =====================================================

# Maps capabilities to their typical "Executive Concern" for automated narrative weighting
CAPABILITY_PRIORITY: Dict[Capability, int] = {
    Capability.DATA_TRUST: 1,   # Foundation: If this is low, nothing else matters
    Capability.QUALITY: 2,      # Risk: Critical for safety and reputation
    Capability.COST: 3,         # Financial: Impact on the bottom line
    Capability.TIME_FLOW: 4,    # Operations: Efficiency and throughput
    Capability.VOLUME: 5,       # Scale: Context for the other metrics
    Capability.VARIANCE: 6,     # Governance: Consistency of execution
    Capability.ACCESS: 7        # Growth: Long-term availability
}

# Define which capabilities are "Risk Indicators" vs "Performance Indicators"
RISK_CAPABILITIES: Set[Capability] = {
    Capability.QUALITY,
    Capability.DATA_TRUST,
    Capability.VARIANCE
}

PERFORMANCE_CAPABILITIES: Set[Capability] = {
    Capability.VOLUME,
    Capability.TIME_FLOW,
    Capability.COST,
    Capability.ACCESS
}

def get_capability_type(cap: Capability) -> str:
    """Categorizes a capability for executive reporting."""
    if cap in RISK_CAPABILITIES:
        return "RISK_GUARDRAIL"
    return "PERFORMANCE_DRIVER"
