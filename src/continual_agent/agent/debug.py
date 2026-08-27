"""Stable, serialisable diagnostics for inspecting neural state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from continual_agent.cognition.affect import AffectiveState
from continual_agent.cognition.readout import Decision

# Debug Snapshot is currently not really used. Keep as simple as possible - could probably
# just directly print properties. Is deepcopy necessary?
@dataclass(frozen=True)
class DebugSnapshot:
    decision: Decision
    affect: AffectiveState
    neural_affect: dict[str, float]
    working_memory: dict[str, object]

    @classmethod
    def from_decision(
        cls,
        decision: Decision,
        affect: AffectiveState,
        neural_affect: dict[str, float] | None = None,
        working_memory: dict[str, object] | None = None,
    ) -> "DebugSnapshot":
        return cls(
            decision=deepcopy(decision),
            affect=deepcopy(affect),
            neural_affect=deepcopy(neural_affect) if neural_affect is not None else {},
            working_memory=deepcopy(working_memory) if working_memory is not None else {},
        )

    def as_dict(self) -> dict[str, object]:
        timed_out = self.decision.timed_out
        return {
            "act": self.decision.action.value,
            "confidence": self.decision.confidence,
            "thinking_ticks": self.decision.ticks,
            "timed_out": timed_out,
            "evidence": {key.value: value for key, value in self.decision.evidence.items()},
            "affect": self.affect.as_dict(),
            "neural_affect": deepcopy(self.neural_affect),
            "working_memory": deepcopy(self.working_memory),
            "control": {
                "committed": not timed_out,
                "emit_token": False,
                "ended": True,
                "interrupted": timed_out,
            },
        }
