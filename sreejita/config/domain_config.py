from dataclasses import dataclass
from typing import Dict


@dataclass
class DomainConfig:
    enabled: bool = True
    min_confidence: float = 0.3
    weight_multiplier: float = 1.0


@dataclass
class DomainEngineConfig:
    default_min_confidence: float = 0.3
    allow_multi_domain: bool = True
    domains: Dict[str, DomainConfig] = None
