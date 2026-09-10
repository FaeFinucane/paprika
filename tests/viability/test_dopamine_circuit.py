"""Multi-seed viability contract for non-learning dopamine TD signals."""

import numpy as np
import pytest
from src.architectures.dopamine import build_dopamine_circuit
from src.diagnostics import inspect_network

from tests.support.perturbations import set_membrane_voltage

pytestmark = pytest.mark.viability


def test_dopamine_td_signs_recover_across_wiring_seeds():
    for seed in range(4):
        _assert_reward_sign(seed, reward=0.7, expected_sign=1, threshold=0.15)
        _assert_reward_sign(seed, reward=-0.7, expected_sign=-1, threshold=0.10)
        _assert_value_signs(seed, "POSITIVE_VALUE", expected_sign=1, threshold=0.25)
        _assert_value_signs(seed, "NEGATIVE_VALUE", expected_sign=-1, threshold=0.18)


def _assert_reward_sign(seed: int, *, reward: float, expected_sign: int, threshold: float) -> None:
    circuit = _settled_circuit(seed)
    comparison = circuit.transition(reward, transition_id="reward")
    response = _tick_values(circuit, 20)
    assert np.sign(comparison.delta) == expected_sign
    assert max(expected_sign * value for value in response) >= threshold
    _assert_recovered_and_healthy(circuit)


def _assert_value_signs(seed: int, population: str, *, expected_sign: int, threshold: float) -> None:
    circuit = _settled_circuit(seed)
    _pulse_value(circuit, population)
    current = circuit.transition(0.0, transition_id="current")
    current_response = _tick_values(circuit, 20)
    assert np.sign(current.delta) == expected_sign
    assert max(expected_sign * value for value in current_response) >= threshold

    previous = circuit.transition(0.0, transition_id="previous")
    previous_response = _tick_values(circuit, 20)
    assert np.sign(previous.delta) == -expected_sign
    assert max(-expected_sign * value for value in previous_response) >= 0.05
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


def _assert_recovered_and_healthy(circuit) -> None:
    settled = _tick_values(circuit, 120)
    assert abs(float(np.mean(settled[-60:]))) <= 0.02
    assert inspect_network(circuit.session.snn).valid


def _tick_values(circuit, ticks: int) -> list[float]:
    values = []
    for _ in range(ticks):
        circuit.tick()
        values.append(circuit.dopamine.value)
    return values
