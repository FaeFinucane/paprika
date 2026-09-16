"""Exploratory curricula for the rate-coded dopamine circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..architectures.dopamine import DopamineCircuit, build_dopamine_circuit


@dataclass(frozen=True)
class TransitionResult:
    reward: float
    dopamine: float
    positive_value: float
    negative_value: float


@dataclass(frozen=True)
class AcquisitionTrial:
    """Trial-aligned observations from one cue-to-reward pairing."""

    index: int
    cue_dopamine_peak: float
    outcome_dopamine_peak: float
    value_positive: float
    value_negative: float
    td_delta: float
    state_to_hidden_strength: float
    hidden_to_positive_value_strength: float


@dataclass
class PairedAcquisitionResult:
    """The live circuit and observations from one exploratory acquisition run."""

    circuit: DopamineCircuit
    trials: list[AcquisitionTrial]


@dataclass(frozen=True)
class FrozenAcquisitionProbes:
    """Read-only checks of a trained cue-to-reward association."""

    cue_dopamine_peak: float
    omission_dopamine_dip: float
    uncued_reward_dopamine_peak: float


def run_transition(
    circuit: DopamineCircuit,
    state_pattern: np.ndarray,
    reward: float,
    *,
    settle_ticks: int = 4,
    transition_id: object | None = None,
) -> TransitionResult:
    """Drive a state, evaluate one boundary, then allow the pulse to settle."""
    circuit.deliver_state(state_pattern, settle_ticks)
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


def run_paired_acquisition(
    seed: int = 0,
    *,
    trials: int = 100,
    settle_ticks: int = 3_000,
    cue_rate: float = 1.0,
    cue_pattern: np.ndarray | None = None,
    value_signal_span: float = 0.17,
    cue_ticks: int = 5,
    cue_to_reward_ticks: int = 5,
    reward: float = 0.7,
    outcome_ticks: int = 10,
    inter_trial_ticks: int = 30,
) -> PairedAcquisitionResult:
    """Run a fixed-delay cue-to-reward acquisition curriculum.

    This is an experiment, not a pass/fail test. It deliberately contains no
    frozen-learning probes: first establish the trajectory produced by paired
    acquisition itself, then use probes to explain that trajectory.
    """
    reward_delays = np.full(trials, cue_to_reward_ticks, dtype=int)
    return _run_acquisition(
        seed,
        reward_delays,
        settle_ticks=settle_ticks,
        cue_rate=cue_rate,
        cue_pattern=cue_pattern,
        value_signal_span=value_signal_span,
        cue_ticks=cue_ticks,
        reward=reward,
        outcome_ticks=outcome_ticks,
        inter_trial_ticks=inter_trial_ticks,
        cycle_delay=cue_to_reward_ticks,
        curriculum_name="paired",
    )


def run_shuffled_acquisition(
    seed: int = 0,
    *,
    trials: int = 100,
    settle_ticks: int = 3_000,
    cue_rate: float = 1.0,
    cue_pattern: np.ndarray | None = None,
    value_signal_span: float = 0.17,
    cue_ticks: int = 5,
    maximum_cue_to_reward_ticks: int = 10,
    reward: float = 0.7,
    outcome_ticks: int = 10,
    inter_trial_ticks: int = 30,
) -> PairedAcquisitionResult:
    """Run the same curriculum with a different cue-to-reward delay each trial."""
    if trials <= 0 or maximum_cue_to_reward_ticks < 0:
        raise ValueError("invalid shuffled-acquisition curriculum")
    delay_rng = np.random.default_rng(seed)
    reward_delays = delay_rng.integers(0, maximum_cue_to_reward_ticks + 1, size=trials)
    return _run_acquisition(
        seed,
        reward_delays,
        settle_ticks=settle_ticks,
        cue_rate=cue_rate,
        cue_pattern=cue_pattern,
        value_signal_span=value_signal_span,
        cue_ticks=cue_ticks,
        reward=reward,
        outcome_ticks=outcome_ticks,
        inter_trial_ticks=inter_trial_ticks,
        cycle_delay=maximum_cue_to_reward_ticks,
        curriculum_name="shuffled",
    )


def _run_acquisition(
    seed: int,
    reward_delays: np.ndarray,
    *,
    settle_ticks: int,
    cue_rate: float,
    cue_pattern: np.ndarray | None,
    value_signal_span: float,
    cue_ticks: int,
    reward: float,
    outcome_ticks: int,
    inter_trial_ticks: int,
    cycle_delay: int,
    curriculum_name: str,
) -> PairedAcquisitionResult:
    trials = reward_delays.size
    if (
        trials <= 0
        or settle_ticks < 0
        or not 0.0 <= cue_rate <= 1.0
        or cue_ticks <= 0
        or np.any(reward_delays < 0)
        or not -1.0 <= reward <= 1.0
        or outcome_ticks <= 0
        or inter_trial_ticks < 0
    ):
        raise ValueError("invalid paired-acquisition curriculum")

    circuit = build_dopamine_circuit(
        seed,
        value_signal_span=value_signal_span,
        enable_dopamine_learning=True,
    )
    pattern = _default_cue_pattern(circuit) if cue_pattern is None else cue_pattern
    _tick(circuit, settle_ticks)
    observations: list[AcquisitionTrial] = []
    for index, reward_delay in enumerate(reward_delays):
        # Episodes are independent. The cue-state transition below supplies
        # the previous-value term for this episode's outcome transition.
        circuit.comparator.reset()
        circuit.deliver_state(cue_rate * pattern, cue_ticks)
        cue_response = _tick(circuit, cue_ticks)
        cue_response.extend(_tick(circuit, int(reward_delay)))

        value_positive = circuit.value_positive_rate.rate
        value_negative = circuit.value_negative_rate.rate
        circuit.transition(
            0.0, transition_id=(curriculum_name, "cue", index), duration=outcome_ticks
        )
        comparison = circuit.transition(
            reward,
            transition_id=(curriculum_name, "outcome", index),
            duration=outcome_ticks,
        )
        outcome_response = _tick(circuit, outcome_ticks)
        _tick(circuit, inter_trial_ticks + cycle_delay - int(reward_delay))

        observations.append(
            AcquisitionTrial(
                index,
                max(cue_response, default=0.0),
                max(outcome_response, default=0.0),
                value_positive,
                value_negative,
                comparison.delta,
                _mean_strength(
                    circuit,
                    (
                        "STATE_to_HIDDEN_ALIGNED",
                        "STATE_to_HIDDEN_OPPOSED",
                        "STATE_to_HIDDEN_NEUTRAL",
                    ),
                ),
                _mean_strength(
                    circuit,
                    (
                        "HIDDEN_ALIGNED_to_POSITIVE_VALUE",
                        "HIDDEN_OPPOSED_to_POSITIVE_VALUE",
                        "HIDDEN_NEUTRAL_to_POSITIVE_VALUE",
                    ),
                ),
            )
        )
    return PairedAcquisitionResult(circuit, observations)


def run_frozen_acquisition_probes(
    circuit: DopamineCircuit,
    *,
    cue_rate: float = 1.0,
    cue_pattern: np.ndarray | None = None,
    cue_ticks: int = 5,
    cue_to_reward_ticks: int = 5,
    outcome_ticks: int = 10,
    settle_ticks: int = 120,
    reward: float = 0.7,
) -> FrozenAcquisitionProbes:
    """Measure cue expectation, omission, and uncued reward without learning."""
    if (
        not 0.0 <= cue_rate <= 1.0
        or cue_ticks <= 0
        or cue_to_reward_ticks < 0
        or outcome_ticks <= 0
        or settle_ticks < 0
        or not -1.0 <= reward <= 1.0
    ):
        raise ValueError("invalid frozen-acquisition probe")

    with circuit.session.frozen_adaptations():
        circuit.reset_episode()
        _tick(circuit, settle_ticks)
        pattern = _default_cue_pattern(circuit) if cue_pattern is None else cue_pattern
        circuit.deliver_state(cue_rate * pattern, cue_ticks)
        cue_response = _tick(circuit, cue_ticks)
        cue_response.extend(_tick(circuit, cue_to_reward_ticks))
        circuit.transition(0.0, transition_id="probe-cue", duration=outcome_ticks)
        circuit.transition(0.0, transition_id="probe-omission", duration=outcome_ticks)
        omission_response = _tick(circuit, outcome_ticks)

        circuit.reset_episode()
        _tick(circuit, settle_ticks)
        circuit.transition(reward, transition_id="probe-uncued-reward", duration=outcome_ticks)
        uncued_reward_response = _tick(circuit, outcome_ticks)

    return FrozenAcquisitionProbes(
        max(cue_response, default=0.0),
        max((-value for value in omission_response), default=0.0),
        max(uncued_reward_response, default=0.0),
    )


def _tick(circuit: DopamineCircuit, ticks: int) -> list[float]:
    values: list[float] = []
    for _ in range(ticks):
        circuit.tick()
        values.append(circuit.dopamine.value)
    return values


def _mean_strength(circuit: DopamineCircuit, projections: tuple[str, ...]) -> float:
    synapses = circuit.session.snn.synapses
    values = [synapses.strength[synapses.projection_mask(projection)] for projection in projections]
    return float(np.mean(np.concatenate(values)))


def _default_cue_pattern(circuit: DopamineCircuit) -> np.ndarray:
    """A sparse example stimulus; production callers should supply an encoder output."""
    pattern = np.zeros(circuit.populations["STATE"].count)
    pattern[: pattern.size // 2] = 1.0
    return pattern


__all__ = [
    "AcquisitionTrial",
    "DopamineCircuit",
    "FrozenAcquisitionProbes",
    "PairedAcquisitionResult",
    "TransitionResult",
    "build_dopamine_circuit",
    "run_frozen_acquisition_probes",
    "run_paired_acquisition",
    "run_shuffled_acquisition",
    "run_transition",
]
