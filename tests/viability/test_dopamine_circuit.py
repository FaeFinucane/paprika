"""Multi-seed behavioural contract for non-learning dopamine signals."""

import numpy as np
import pytest
from src.architectures.dopamine import build_dopamine_circuit
from src.diagnostics import inspect_network

from tests.support.perturbations import clear_transient_activity, set_membrane_voltage

pytestmark = pytest.mark.viability


def test_dopamine_has_a_quiet_tonic_baseline_across_wiring_seeds():
    for seed in range(4):
        circuit = _settled_circuit(seed)
        rates: list[float] = []
        modulation: list[float] = []
        for _ in range(240):
            circuit.tick()
            rates.append(circuit.dopamine_rate.rate)
            modulation.append(circuit.dopamine.value)

        assert 0.22 <= float(np.mean(rates)) <= 0.28
        assert float(np.mean(np.abs(modulation))) <= 0.005
        assert max(abs(value) for value in modulation) <= 0.01
        assert inspect_network(circuit.session.snn).valid


def test_dopamine_comparator_has_signed_responses_and_recovers_across_wiring_seeds():
    for seed in range(4):
        _assert_reward_sign(seed, reward=0.7, expected_sign=1, threshold=0.14)
        _assert_reward_sign(seed, reward=-0.7, expected_sign=-1, threshold=0.15)
        _assert_value_signs(
            seed, "POSITIVE_VALUE", expected_sign=1, threshold=0.17, prior_threshold=0.10
        )
        _assert_value_signs(
            seed, "NEGATIVE_VALUE", expected_sign=-1, threshold=0.13, prior_threshold=0.035
        )


def test_dopamine_reward_pulses_have_a_resolved_signal_to_noise_ratio_across_seeds():
    """A sparse reward code must be visible without excessive pulse variation."""
    for seed in range(4):
        circuit = _settled_circuit(seed)
        with circuit.session.frozen_adaptations():
            baseline = _tick_values(circuit, 120)
            peaks: list[float] = []
            for trial in range(12):
                circuit.comparator.reset()
                circuit.transition(0.7, transition_id=("signal-noise", trial))
                peaks.append(max(_tick_values(circuit, 20)))
                _tick_values(circuit, 100)

        peak_mean = float(np.mean(peaks))
        peak_std = float(np.std(peaks))
        baseline_std = float(np.std(baseline))
        assert peak_mean >= 0.10
        assert peak_mean >= 10.0 * baseline_std
        assert peak_std / peak_mean <= 0.50
        assert inspect_network(circuit.session.snn).valid


def test_dopamine_cue_path_carries_distinct_states_to_value_across_wiring_seeds():
    """Frozen state patterns reach value without a reward or comparator pulse."""
    for seed in range(4):
        circuit = _settled_circuit(seed)
        pattern_a = _state_pattern(circuit, 0)
        pattern_b = _state_pattern(circuit, circuit.populations["STATE"].count // 2)
        with circuit.session.frozen_adaptations():
            baseline_hidden, baseline_value = _cue_path_response(circuit, None)
            hidden_a, value_a = _cue_path_response(circuit, pattern_a)
            hidden_b, value_b = _cue_path_response(circuit, pattern_b)

        hidden_effect = min(
            np.linalg.norm(hidden_a - baseline_hidden),
            np.linalg.norm(hidden_b - baseline_hidden),
        )
        value_effect = min(
            np.linalg.norm(value_a - baseline_value),
            np.linalg.norm(value_b - baseline_value),
        )
        hidden_separation = np.linalg.norm(hidden_a - hidden_b)
        value_separation = np.linalg.norm(value_a - value_b)

        # The cue route must reach both stages and preserve a distinction; it
        # must not merely raise every value neuron to a saturated common rate.
        assert hidden_effect >= 0.10
        assert hidden_separation >= 0.10
        assert value_effect >= 0.05
        assert value_separation >= 0.05
        assert max(float(np.mean(value_a)), float(np.mean(value_b))) <= 0.20
        assert inspect_network(circuit.session.snn).valid


def _assert_reward_sign(seed: int, *, reward: float, expected_sign: int, threshold: float) -> None:
    circuit = _settled_circuit(seed)
    comparison = circuit.transition(reward, transition_id="reward")
    response = _tick_values(circuit, 20)
    assert np.sign(comparison.delta) == expected_sign
    assert max(expected_sign * value for value in response) >= threshold
    _assert_recovered_and_healthy(circuit)


def _assert_value_signs(
    seed: int,
    population: str,
    *,
    expected_sign: int,
    threshold: float,
    prior_threshold: float,
) -> None:
    circuit = _settled_circuit(seed)
    _pulse_value(circuit, population)
    current = circuit.transition(0.0, transition_id="current")
    current_response = _tick_values(circuit, 20)
    assert np.sign(current.delta) == expected_sign
    assert max(expected_sign * value for value in current_response) >= threshold

    previous = circuit.transition(0.0, transition_id="previous")
    previous_response = _tick_values(circuit, 20)
    assert np.sign(previous.delta) == -expected_sign
    assert max(-expected_sign * value for value in previous_response) >= prior_threshold
    _assert_recovered_and_healthy(circuit)


def _settled_circuit(seed: int):
    circuit = build_dopamine_circuit(seed, enable_dopamine_learning=False)
    _tick_values(circuit, 240)
    return circuit


def _pulse_value(circuit, population: str) -> None:
    value_population = circuit.populations[population]
    for _ in range(3):
        set_membrane_voltage(circuit.session.snn, value_population, voltage=3.0)
        circuit.tick()


def _cue_path_response(circuit, pattern: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return raw hidden and positive-value response vectors for one input pattern."""
    clear_transient_activity(circuit.session.snn)
    if pattern is not None:
        circuit.deliver_state(pattern, duration=8)
    hidden_samples: list[np.ndarray] = []
    value_samples: list[np.ndarray] = []
    for _ in range(12):
        spikes = circuit.tick()
        hidden_samples.append(
            np.concatenate(
                [
                    spikes.population(circuit.populations[name]).astype(float)
                    for name in ("HIDDEN_ALIGNED", "HIDDEN_OPPOSED", "HIDDEN_NEUTRAL")
                ]
            )
        )
        value_samples.append(spikes.population(circuit.populations["POSITIVE_VALUE"]).astype(float))
    return np.mean(hidden_samples, axis=0), np.mean(value_samples, axis=0)


def _state_pattern(circuit, start: int) -> np.ndarray:
    """Create one equal-energy sparse test pattern on the generic STATE surface."""
    pattern = np.zeros(circuit.populations["STATE"].count)
    pattern[start : start + pattern.size // 2] = 1.0
    return pattern


def _assert_recovered_and_healthy(circuit) -> None:
    settled = _tick_values(circuit, 180)
    tail = settled[-60:]
    assert float(np.mean(np.abs(tail))) <= 0.005
    assert max(abs(value) for value in tail) <= 0.02
    assert inspect_network(circuit.session.snn).valid


def _tick_values(circuit, ticks: int) -> list[float]:
    values = []
    for _ in range(ticks):
        circuit.tick()
        values.append(circuit.dopamine.value)
    return values
