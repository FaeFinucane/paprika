from dataclasses import dataclass

from .evaluation import EvaluationReport


@dataclass(frozen=True)
class RewardPolicy:
    correct: float = 1.0
    structured: float = 0.2
    malformed: float = -0.2

    def reward(self, report: EvaluationReport) -> float:
        if report.score == 1.0:
            return self.correct
        if report.matched and not report.unwanted:
            return self.structured
        return self.malformed
