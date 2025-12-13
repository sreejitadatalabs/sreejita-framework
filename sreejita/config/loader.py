import yaml
from .defaults import DEFAULT_CONFIG

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        user_config = yaml.safe_load(f)

    # merge defaults
    config = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict):
            config[key] = {**config.get(key, {}), **value}
        else:
            config[key] = value
    return config
