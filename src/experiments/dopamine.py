"""Exploratory cue-to-outcome curricula for the VTA circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..builder import NetworkBuilder
from ..circuits.asymmetric_recurrent import AsymmetricRecurrentSpec
from ..circuits.vta_dopamine import DopamineCircuit, add_vta_dopamine_circuit


@dataclass(frozen=True)
class AcquisitionTrial:
    """Event-aligned activity from one continuous cue-to-outcome trial."""

    index: int
    cue_dopamine_peak: float
    outcome_dopamine_peak: float
    vta_inhibitory_peak: float
    inferred_state_rate: float
    temporal_rate: float


@dataclass
class PairedAcquisitionResult:
    circuit: DopamineCircuit
    trials: list[AcquisitionTrial]


def run_paired_acquisition(
    seed: int = 0,
    *,
    trials: int = 100,
    settle_ticks: int = 1_000,
    cue_pattern: np.ndarray | None = None,
    cue_ticks: int = 5,
    cue_to_outcome_ticks: int = 5,
    outcome: float = 0.7,
    outcome_ticks: int = 10,
    inter_trial_ticks: int = 30,
    temporal: AsymmetricRecurrentSpec = AsymmetricRecurrentSpec(),
) -> PairedAcquisitionResult:
    """Run repeated cue/outcome pairings without resetting neural state."""
    if trials <= 0 or cue_ticks <= 0 or cue_to_outcome_ticks < 0 or outcome_ticks <= 0:
        raise ValueError("invalid cue-to-outcome timing")
    if settle_ticks < 0 or inter_trial_ticks < 0 or not -1.0 <= outcome <= 1.0:
        raise ValueError("invalid acquisition configuration")
    circuit = add_vta_dopamine_circuit(NetworkBuilder(), temporal=temporal).build(seed)
    pattern = _default_cue_pattern(circuit) if cue_pattern is None else cue_pattern
    _tick(circuit, settle_ticks)
    observations: list[AcquisitionTrial] = []
    for index in range(trials):
        circuit.present_cue(pattern, cue_ticks)
        cue_response, cue_inhibitory, cue_state, cue_temporal = _observe(
            circuit, cue_ticks + cue_to_outcome_ticks
        )
        circuit.deliver_outcome(outcome, outcome_ticks)
        outcome_response, outcome_inhibitory, _, _ = _observe(circuit, outcome_ticks)
        observations.append(
            AcquisitionTrial(
                index,
                max(cue_response, default=0.0),
                max(outcome_response, default=0.0),
                max((*cue_inhibitory, *outcome_inhibitory), default=0.0),
                cue_state,
                cue_temporal,
            )
        )
        _tick(circuit, inter_trial_ticks)
    return PairedAcquisitionResult(circuit, observations)


def run_shuffled_acquisition(
    seed: int = 0,
    *,
    trials: int = 100,
    settle_ticks: int = 1_000,
    cue_pattern: np.ndarray | None = None,
    cue_ticks: int = 5,
    maximum_cue_to_outcome_ticks: int = 10,
    outcome: float = 0.7,
    outcome_ticks: int = 10,
    inter_trial_ticks: int = 30,
    temporal: AsymmetricRecurrentSpec = AsymmetricRecurrentSpec(),
) -> PairedAcquisitionResult:
    """Run cue/outcome pairings with a sampled delay on each continuous trial."""
    if maximum_cue_to_outcome_ticks < 0:
        raise ValueError("maximum cue-to-outcome delay must be non-negative")
    if trials <= 0:
        raise ValueError("trials must be positive")
    delays = np.random.default_rng(seed).integers(0, maximum_cue_to_outcome_ticks + 1, trials)
    circuit = add_vta_dopamine_circuit(NetworkBuilder(), temporal=temporal).build(seed)
    if settle_ticks < 0 or cue_ticks <= 0 or outcome_ticks <= 0 or inter_trial_ticks < 0:
        raise ValueError("invalid acquisition timing")
    if not -1.0 <= outcome <= 1.0:
        raise ValueError("outcome must be in [-1, 1]")
    pattern = _default_cue_pattern(circuit) if cue_pattern is None else cue_pattern
    _tick(circuit, settle_ticks)
    observations: list[AcquisitionTrial] = []
    for index, delay in enumerate(delays):
        circuit.present_cue(pattern, cue_ticks)
        cue_response, cue_inhibitory, cue_state, cue_temporal = _observe(
            circuit, cue_ticks + int(delay)
        )
        circuit.deliver_outcome(outcome, outcome_ticks)
        outcome_response, outcome_inhibitory, _, _ = _observe(circuit, outcome_ticks)
        observations.append(
            AcquisitionTrial(
                index,
                max(cue_response, default=0.0),
                max(outcome_response, default=0.0),
                max((*cue_inhibitory, *outcome_inhibitory), default=0.0),
                cue_state,
                cue_temporal,
            )
        )
        _tick(circuit, inter_trial_ticks)
    return PairedAcquisitionResult(circuit, observations)


def _observe(circuit: DopamineCircuit, ticks: int) -> tuple[list[float], list[float], float, float]:
    dopamine: list[float] = []
    inhibitory: list[float] = []
    state: list[float] = []
    temporal: list[float] = []
    for _ in range(ticks):
        spikes = circuit.tick()
        dopamine.append(circuit.dopamine.value)
        inhibitory.append(float(np.mean(spikes.population(circuit.populations["VTA_INHIB"]))))
        state.append(
            float(
                np.mean(spikes.population(circuit.populations[circuit.inferred_state.excitatory]))
            )
        )
        temporal.append(
            float(np.mean(spikes.population(circuit.populations[circuit.temporal.excitatory])))
        )
    return dopamine, inhibitory, float(np.mean(state)), float(np.mean(temporal))


def _tick(circuit: DopamineCircuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()


def _default_cue_pattern(circuit: DopamineCircuit) -> np.ndarray:
    pattern = np.zeros(circuit.populations["CUE"].count)
    pattern[: pattern.size // 2] = 1.0
    return pattern


__all__ = [
    "AcquisitionTrial",
    "DopamineCircuit",
    "PairedAcquisitionResult",
    "run_paired_acquisition",
    "run_shuffled_acquisition",
]
