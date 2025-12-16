from abc import ABC, abstractmethod
from sreejita.core.decision import DecisionExplanation


class DecisionObserver(ABC):
    @abstractmethod
    def record(self, decision: DecisionExplanation):
        pass
