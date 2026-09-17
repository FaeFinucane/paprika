"""Contract for generic network tracing."""

import pytest

from src.builder import NetworkBuilder
from src.diagnostics import NetworkTrace
from src.interaction.drives import TonicDrive
from src.network.connectivity import FanOutSpec, StrengthSpec
from src.session import Session

pytestmark = pytest.mark.unit


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
