from dataclasses import dataclass, field
from typing import Dict


# -------------------------------------------------
# PER-DOMAIN CONFIG
# -------------------------------------------------
@dataclass
class DomainConfig:
    """
    Configuration for a single domain engine.
    """
    enabled: bool = True
    min_confidence: float = 0.3
    weight_multiplier: float = 1.0


# -------------------------------------------------
# DOMAIN ENGINE CONFIG (v3.3 SAFE)
# -------------------------------------------------
@dataclass
class DomainEngineConfig:
    """
    Global domain engine configuration (v3.3 SAFE)
    """
    
    # v3.3 rules:
    # - domains MUST always be a dict
    # - no shared mutable defaults
    
    default_min_confidence: float = 0.3
    allow_multi_domain: bool = True
    domains: Dict[str, DomainConfig] = field(
        default_factory=lambda: {
            "healthcare": DomainConfig(
                enabled=True,
                min_confidence=0.35,  # CRITICAL: Must match HealthcareDomainDetector safety floor
                weight_multiplier=1.0,
            ),
            "retail": DomainConfig(
                enabled=True,
                min_confidence=0.3,
                weight_multiplier=1.0,
            ),
            "generic": DomainConfig(
                enabled=True,
                min_confidence=0.0,  # Generic domain always accepts
                weight_multiplier=1.0,
            ),
        }
    )

    # --------------------------
    # SAFE ACCESSORS
    # --------------------------

    def get_domain_config(self, domain: str) -> DomainConfig:
        """
        Returns domain-specific config or a safe default.
        """
        return self.domains.get(
            domain,
            DomainConfig(min_confidence=self.default_min_confidence),
        )

    def is_domain_enabled(self, domain: str) -> bool:
        return self.get_domain_config(domain).enabled

    def get_min_confidence(self, domain: str) -> float:
        """Get minimum confidence threshold for a domain."""
        return self.get_domain_config(domain).min_confidence
