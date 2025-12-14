import yaml
import copy
from pathlib import Path

from .defaults import DEFAULT_CONFIG


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
