from dataclasses import dataclass

from .conversation import Turn


@dataclass(frozen=True)
class EvaluationReport:
    identity: str
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    matched: int
    missing: tuple[str, ...]
    unwanted: tuple[str, ...]
    score: float
    status: str


class Evaluator:
    def evaluate(self, turn: Turn, expected: tuple[str, ...], identity: str | None = None):
        exp = tuple(expected)
        obs = tuple(x.feature for x in turn.outputs)
        matched = sum(a == b for a, b in zip(exp, obs))
        missing = exp[matched:]
        unwanted = obs[matched:]
        score = matched / max(1, len(exp))
        if turn.status == "eos" and not missing and not unwanted:
            score = 1.0
        return EvaluationReport(
            identity or str(id(turn)),
            exp,
            obs,
            matched,
            missing,
            unwanted,
            float(score),
            turn.status,
        )
