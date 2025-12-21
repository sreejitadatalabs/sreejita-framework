import yaml
import copy
from pathlib import Path

from .defaults import DEFAULT_CONFIG
from sreejita.config.domain_config import DomainEngineConfig, DomainConfig


# -------------------------------------------------
# DOMAIN ENGINE CONFIG LOADER
# -------------------------------------------------
def load_domain_engine_config(cfg: dict) -> DomainEngineConfig:
    domains_cfg = {}

    for domain, values in cfg.get("domains", {}).items():
        domains_cfg[domain] = DomainConfig(**values)

    return DomainEngineConfig(
        default_min_confidence=cfg.get("default_min_confidence", 0.3),
        allow_multi_domain=cfg.get("allow_multi_domain", True),
        domains=domains_cfg,
    )


# -------------------------------------------------
# MAIN CONFIG LOADER (v3.3 SAFE)
# -------------------------------------------------
def load_config(path: str | None) -> dict:
    """
    Load and merge user config with framework defaults.

    v3.3 rules:
    - Defaults must ALWAYS win if user omits fields
    - dataset / domain sections are OPTIONAL
    - output_dir MUST always exist
    """

    # -------------------------------------------------
    # 1️⃣ Load user config (if provided)
    # -------------------------------------------------
    user_config = {}

    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        if not isinstance(user_config, dict):
            raise ValueError("Config file must contain a YAML dictionary")

    # -------------------------------------------------
    # 2️⃣ Merge with defaults (SAFE)
    # -------------------------------------------------
    config = copy.deepcopy(DEFAULT_CONFIG)

    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value

    # -------------------------------------------------
    # 3️⃣ Enforce REQUIRED v3.3 invariants
    # -------------------------------------------------

    # ALWAYS guarantee output_dir
    config.setdefault("output_dir", "runs")

    # ALWAYS guarantee metadata dict
    config.setdefault("metadata", {})

    # ALWAYS guarantee export_pdf flag
    config.setdefault("export_pdf", False)

    # -------------------------------------------------
    # 4️⃣ Attach domain engine config (if present)
    # -------------------------------------------------
    if "domains" in config or "default_min_confidence" in config:
        config["domain_engine"] = load_domain_engine_config(config)

    # -------------------------------------------------
    # 5️⃣ FINAL RETURN
    # -------------------------------------------------
    return config
