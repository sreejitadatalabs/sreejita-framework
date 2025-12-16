from sreejita.observability.hooks import (
    ConsoleDecisionObserver,
    FileDecisionObserver,
    DecisionObserver
)

def build_observers(config: dict) -> list[DecisionObserver]:
    observers = []

    for obs in config.get("observers", []):
        if obs["type"] == "console":
            observers.append(ConsoleDecisionObserver())

        elif obs["type"] == "file":
            observers.append(
                FileDecisionObserver(path=obs.get("path", "decision_audit.jsonl"))
            )

    return observers
