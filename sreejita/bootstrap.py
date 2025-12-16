from sreejita.domains.router import register_observer
from sreejita.observability.factory import build_observers

def bootstrap_decision_engine(config: dict):
    observers = build_observers(config.get("decision_engine", {}))

    for observer in observers:
        register_observer(observer)
