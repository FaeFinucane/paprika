import pytest
from src.builder import NetworkBuilder, TonicDriveSpec
from src.diagnostics import NetworkTrace, inspect_network
from src.interaction.drives import TonicDrive
from src.network.connectivity import FanOutSpec, StrengthSpec
from src.session import Session

pytestmark = pytest.mark.unit


def test_network_builder_compiles_a_reusable_basic_session():
    builder = NetworkBuilder()
    builder.add_population("SOURCE", 2, plugins=(TonicDriveSpec(0.2),))
    builder.add_population("TARGET", 2)
    builder.connect("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2))
    session = builder.compile(1)
    snn = session.snn
    assert snn.layout.population("SOURCE").count == 2
    assert any(isinstance(hook, TonicDrive) for hook in session.drive_sources)
    assert inspect_network(snn).valid


def test_network_trace_records_generic_population_and_projection_state():
    builder = NetworkBuilder()
    builder.add_population("SOURCE", 1)
    builder.add_population("TARGET", 1)
    builder.connect("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2))
    snn = builder.compile(2).snn
    source_drive = TonicDrive(snn.layout.population("SOURCE"), 1.0)
    trace = NetworkTrace(snn, current_drives=[source_drive])
    session = Session.build(snn, [source_drive, trace])
    session.tick()
    assert len(trace.samples) == 1
    sample = trace.samples[0]
    assert set(sample.population_rates) == {"SOURCE", "TARGET"}
    assert sample.population_drive_currents == {"SOURCE": 1.0, "TARGET": 0.0}
    assert "SOURCE_to_TARGET" in sample.projection_strengths
