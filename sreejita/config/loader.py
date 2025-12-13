import yaml
from .defaults import DEFAULT_CONFIG

def load_config(path: str) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f)

    config = DEFAULT_CONFIG.copy()
    for k, v in user.items():
        if isinstance(v, dict):
            config[k] = {**config.get(k, {}), **v}
        else:
            config[k] = v
    return config
