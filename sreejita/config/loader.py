import yaml
import copy
from pathlib import Path
from typing import Optional

from .defaults import DEFAULT_CONFIG
from sreejita.config.domain_config import DomainEngineConfig, DomainConfig


# -------------------------------------------------
# DOMAIN ENGINE CONFIG LOADER (OPTIONAL / ADVANCED)
# -------------------------------------------------
def load_domain_engine_config(cfg: dict) -> DomainEngineConfig:
    """
    Load domain engine configuration safely.

    v3.3 rules:
    - Missing domains are allowed
    - Partial domain configs are allowed
    - Never mutates input
    """

    domains_cfg = {}

    for domain, values in (cfg.get("domains") or {}).items():
        if isinstance(values, dict):
            domains_cfg[domain] = DomainConfig(**values)

    return DomainEngineConfig(
        default_min_confidence=cfg.get("default_min_confidence", 0.3),
        allow_multi_domain=cfg.get("allow_multi_domain", True),
        domains=domains_cfg,
    )


# -------------------------------------------------
# MAIN CONFIG LOADER (AUTHORITATIVE)
# -------------------------------------------------
def load_config(path: Optional[str]) -> dict:
    """
    Load and merge user config with framework defaults.

    v3.3 SAFE:
    - Python 3.9 compatible
    - No PEP 604 union types
    - No shared mutable defaults
    - Backward compatible with v1.x / v2.x
    """

    # ---------------------------------------------
    # No config provided → return clean defaults
    # ---------------------------------------------
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    # ---------------------------------------------
    # Start from a clean copy of defaults
    # ---------------------------------------------
    config = copy.deepcopy(DEFAULT_CONFIG)

    # ---------------------------------------------
    # Merge user overrides (safe shallow merge)
    # ---------------------------------------------
    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and isinstance(config.get(key), dict)
        ):
            config[key].update(value)
        else:
            config[key] = value

    # ---------------------------------------------
    # Minimal validation (v1.x compatibility)
    # ---------------------------------------------
    if "dataset" not in config:
        raise ValueError("Missing required 'dataset' section in config")

    if "domain" not in config:
        raise ValueError("Missing required 'domain' section in config")

    return config
