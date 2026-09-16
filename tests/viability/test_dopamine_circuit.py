"""Behavioural contract for the cue/state/temporal VTA circuit."""

import numpy as np
import pytest
from src.architectures.dopamine import build_dopamine_circuit
from src.diagnostics import inspect_network

from tests.support.perturbations import set_membrane_voltage

pytestmark = pytest.mark.viability


def test_vta_circuit_has_tonic_da_and_signed_outcome_responses_across_seeds():
    for seed in range(4):
        circuit = build_dopamine_circuit(seed, enable_dopamine_learning=False)
        _tick(circuit, 240)
        baseline = _modulation(circuit, 60)
        assert abs(float(np.mean(baseline))) <= 0.02

        circuit.deliver_outcome(0.7, 10)
        positive = _modulation(circuit, 30)
        circuit.deliver_outcome(-0.7, 10)
        negative = _modulation(circuit, 30)

        assert max(positive) >= 0.20
        assert min(negative) <= -0.15
        assert inspect_network(circuit.session.snn).valid


def test_cue_drives_a_distinct_inferred_state_without_an_episode_reset():
    circuit = build_dopamine_circuit(1, enable_dopamine_learning=False)
    _tick(circuit, 240)
    cue = np.zeros(circuit.populations["CUE"].count)
    cue[: cue.size // 2] = 1.0
    circuit.present_cue(cue, 12)
    state_rates = []
    temporal_rates = []
    inhibitory_rates = []
    for _ in range(24):
        spikes = circuit.tick()
        state_rates.append(
            float(
                np.mean(spikes.population(circuit.populations[circuit.inferred_state.excitatory]))
            )
        )
        temporal_rates.append(
            max(
                float(np.mean(spikes.population(circuit.populations[stage])))
                for stage in circuit.temporal.stages
            )
        )
        inhibitory_rates.append(float(np.mean(spikes.population(circuit.populations["VTA_INHIB"]))))

    assert max(state_rates) >= 0.25
    assert max(temporal_rates) >= 0.10
    assert max(inhibitory_rates) >= 0.10
    assert circuit.inferred_state.excitatory == "INFERRED_STATE_E"
    assert circuit.temporal.stages[0] == "TEMPORAL_0"
    assert set(circuit.populations) >= {"CUE", "VTA_DA", "VTA_INHIB"}


def test_vta_inhibition_suppresses_da_through_an_ordinary_learnable_projection():
    circuit = build_dopamine_circuit(2, enable_dopamine_learning=False)
    _tick(circuit, 240)
    synapses = circuit.session.snn.synapses
    mask = synapses.projection_mask("VTA_INHIB_to_VTA_DA")
    assert mask.any()
    assert set(synapses.learning[mask]) == {"inhibitory_homeostatic"}

    set_membrane_voltage(circuit.session.snn, circuit.populations["VTA_INHIB"], voltage=3.0)
    response = _modulation(circuit, 20)
    assert min(response) <= -0.08


def _tick(circuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()


def _modulation(circuit, ticks: int) -> list[float]:
    values = []
    for _ in range(ticks):
        circuit.tick()
        values.append(circuit.dopamine.value)
    return values
