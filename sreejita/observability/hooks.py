from abc import ABC, abstractmethod
from sreejita.core.decision import DecisionExplanation
import json
from pathlib import Path


class DecisionObserver(ABC):
    @abstractmethod
    def record(self, decision: DecisionExplanation):
        pass


class ConsoleDecisionObserver(DecisionObserver):
    def record(self, decision: DecisionExplanation):
        print("[SREEJITA DECISION]")
        print(decision)


class FileDecisionObserver(DecisionObserver):
    def __init__(self, path: str = "decision_audit.jsonl"):
        self.path = Path(path)

    def record(self, decision: DecisionExplanation):
        with self.path.open("a") as f:
            f.write(json.dumps(decision.__dict__) + "\n")
