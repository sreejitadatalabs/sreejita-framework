import yaml
import copy
from pathlib import Path

from .defaults import DEFAULT_CONFIG
from sreejita.config.domain_config import DomainEngineConfig, DomainConfig

def load_domain_engine_config(cfg: dict) -> DomainEngineConfig:
    domains_cfg = {}

    for domain, values in cfg.get("domains", {}).items():
        domains_cfg[domain] = DomainConfig(**values)

    return DomainEngineConfig(
        default_min_confidence=cfg.get("default_min_confidence", 0.3),
        allow_multi_domain=cfg.get("allow_multi_domain", True),
        domains=domains_cfg
    )


def load_config(path: str) -> dict:
    """
    Load and merge user config with framework defaults.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    # Deep copy defaults to avoid cross-run mutation
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Shallow merge user overrides
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value

    # Minimal validation (v1.7 scope)
    if "dataset" not in config:
        raise ValueError("Missing required 'dataset' section in config")

    if "domain" not in config:
        raise ValueError("Missing required 'domain' section in config")

    return config
