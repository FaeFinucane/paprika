"""Small deterministic entry point for the rate-coded dopamine circuit."""

from __future__ import annotations

from dataclasses import dataclass

from ..architectures.dopamine import DopamineCircuit, build_dopamine_circuit


@dataclass(frozen=True)
class TransitionResult:
    reward: float
    dopamine: float
    positive_value: float
    negative_value: float


def run_transition(
    circuit: DopamineCircuit,
    state_rate: float,
    reward: float,
    *,
    settle_ticks: int = 4,
    transition_id: object | None = None,
) -> TransitionResult:
    """Drive a state, evaluate one boundary, then allow the pulse to settle."""
    circuit.state_input.write(state_rate, settle_ticks)
    for _ in range(settle_ticks):
        circuit.tick()
    circuit.transition(reward, transition_id=transition_id)
    for _ in range(settle_ticks):
        circuit.tick()
    return TransitionResult(
        reward,
        circuit.dopamine.value,
        circuit.value_positive_rate.rate,
        circuit.value_negative_rate.rate,
    )


__all__ = ["DopamineCircuit", "TransitionResult", "build_dopamine_circuit", "run_transition"]
