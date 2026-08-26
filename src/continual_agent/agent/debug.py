"""Stable, serialisable diagnostics for inspecting neural state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from continual_agent.cognition.affect import AffectiveState
from continual_agent.cognition.readout import Decision


@dataclass(frozen=True)
class DebugSnapshot:
    act: str
    confidence: float
    thinking_ticks: int
    timed_out: bool
    evidence: dict[str, float]
    affect: dict[str, float]
    neural_affect: dict[str, float]
    working_memory: dict[str, object]
    control: dict[str, bool]

    @classmethod
    def from_decision(
        cls,
        decision: Decision,
        affect: AffectiveState,
        neural_affect: dict[str, float] | None = None,
        working_memory: dict[str, object] | None = None,
    ) -> "DebugSnapshot":
        timed_out = decision.timed_out
        return cls(
            act=decision.action.value,
            confidence=decision.confidence,
            thinking_ticks=decision.ticks,
            timed_out=timed_out,
            evidence={key.value: value for key, value in decision.evidence.items()},
            affect=affect.as_dict(),
            neural_affect=neural_affect or {},
            working_memory=working_memory or {},
            control={
                "committed": not timed_out,
                "emit_token": False,
                "ended": True,
                "interrupted": timed_out,
            },
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
