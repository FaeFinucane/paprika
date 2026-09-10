import numpy as np
import pytest
from src.diagnostics import NetworkTrace, inspect_network
from src.interaction.drives import TonicDrive
from src.network.connectivity import ConnectionSpec, FanOutSpec, StrengthSpec
from src.network.definition import NetworkDefinition
from src.network.population import NeuronPopulationSpec
from src.session import Session

pytestmark = pytest.mark.unit


def test_network_definition_compiles_a_reusable_basic_network():
    definition = NetworkDefinition(
        (
            NeuronPopulationSpec("SOURCE", 2),
            NeuronPopulationSpec("TARGET", 2),
        ),
        (ConnectionSpec("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2)),),
    )
    snn = definition.compile(np.random.default_rng(1))
    assert snn.layout.population("SOURCE").count == 2
    assert inspect_network(snn).valid


def test_network_trace_records_generic_population_and_projection_state():
    definition = NetworkDefinition(
        (NeuronPopulationSpec("SOURCE", 1), NeuronPopulationSpec("TARGET", 1)),
        (ConnectionSpec("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2)),),
    )
    snn = definition.compile(np.random.default_rng(2))
    source_drive = TonicDrive(snn.layout.population("SOURCE"), 1.0)
    trace = NetworkTrace(snn, current_drives=[source_drive])
    session = Session.build(snn, [source_drive, trace])
    session.tick()
    assert len(trace.samples) == 1
    sample = trace.samples[0]
    assert set(sample.population_rates) == {"SOURCE", "TARGET"}
    assert sample.population_drive_currents == {"SOURCE": 1.0, "TARGET": 0.0}
    assert "SOURCE_to_TARGET" in sample.projection_strengths
