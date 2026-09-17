"""Viability envelope for the complete VTA dopamine circuit."""

import numpy as np
import pytest
from tests.support.perturbations import set_membrane_voltage

from src.builder import NetworkBuilder
from src.diagnostics import inspect_network
from src.network.connectivity import FanInSpec

from .vta_dopamine import add_vta_dopamine_circuit, build_dopamine_circuit

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


def test_dopamine_learning_selects_a_later_vta_inhibitory_temporal_readout():
    early_traces = []
    late_traces = []
    for seed in range(4):
        circuit = build_dopamine_circuit(seed)
        _tick(circuit, 1_000)
        traces = _paired_vta_inhibitory_traces(circuit, trials=80)
        early_traces.extend(traces[:20])
        late_traces.extend(traces[-20:])

    early_peak = int(np.argmax(np.mean(early_traces, axis=0)))
    late_peak = int(np.argmax(np.mean(late_traces, axis=0)))
    assert early_peak <= 8
    assert late_peak >= 9
    assert late_peak > early_peak


def test_vta_inhibition_suppresses_da_through_an_ordinary_learnable_projection():
    circuit = build_dopamine_circuit(2, enable_dopamine_learning=False)
    _tick(circuit, 240)
    synapses = circuit.session.snn.synapses
    mask = synapses.projection_mask("VTA_INHIB_to_VTA_DA")
    assert mask.any()
    assert set(synapses.learning[mask]) == {"dopamine_stdp"}

    set_membrane_voltage(circuit.session.snn, circuit.populations["VTA_INHIB"], voltage=3.0)
    assert min(_modulation(circuit, 20)) <= -0.06


def test_vta_temporal_readout_can_be_edited_through_declared_population_handles():
    builder = NetworkBuilder()
    declaration = add_vta_dopamine_circuit(builder)
    projection = builder.connection(declaration.temporal.excitatory, declaration.vta_inhibitory)
    projection.topology = FanInSpec(24)
    projection.strength.mean = 0.8 / 24

    circuit = declaration.build(1, enable_dopamine_learning=False)
    mask = circuit.session.snn.synapses.projection_mask("TEMPORAL_E_to_VTA_INHIB")
    assert mask.sum() > 250
    assert projection.strength.mean == 0.8 / 24


def _tick(circuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()


def _modulation(circuit, ticks: int) -> list[float]:
    values = []
    for _ in range(ticks):
        circuit.tick()
        values.append(circuit.dopamine.value)
    return values


def _paired_vta_inhibitory_traces(circuit, *, trials: int) -> list[list[float]]:
    cue = np.zeros(circuit.populations["CUE"].count)
    cue[: cue.size // 2] = 1.0
    traces = []
    for _ in range(trials):
        circuit.present_cue(cue, 5)
        trace = []
        for _ in range(10):
            spikes = circuit.tick()
            trace.append(float(np.mean(spikes.population(circuit.populations["VTA_INHIB"]))))
        circuit.deliver_outcome(0.7, 10)
        for _ in range(10):
            spikes = circuit.tick()
            trace.append(float(np.mean(spikes.population(circuit.populations["VTA_INHIB"]))))
        traces.append(trace)
        _tick(circuit, 30)
    return traces
