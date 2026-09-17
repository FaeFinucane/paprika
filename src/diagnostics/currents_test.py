"""Contract for projection-current diagnostics."""

from src.builder import NetworkBuilder
from src.diagnostics import incoming_projection_currents
from src.network.connectivity import FanOutSpec, StrengthSpec


def test_current_attribution_preserves_source_transmitter_sign():
    builder = NetworkBuilder()
    builder.add_population("EXCITATORY", 1)
    builder.add_population("INHIBITORY", 1, output="inhibitory")
    builder.add_population("TARGET", 1)
    builder.connect("EXCITATORY", "TARGET", FanOutSpec(1), StrengthSpec(0.3))
    builder.connect("INHIBITORY", "TARGET", FanOutSpec(1), StrengthSpec(0.2))
    snn = builder.compile(1).snn
    snn.neurons.voltage[:] = 3.0
    spikes = snn.step()

    currents = incoming_projection_currents(snn, spikes, snn.layout.population("TARGET"))

    assert currents == {"EXCITATORY_to_TARGET": 0.3, "INHIBITORY_to_TARGET": -0.2}
