import numpy as np
from src.diagnostics import incoming_projection_currents
from src.network.connectivity import ConnectionSpec, FanOutSpec, StrengthSpec
from src.network.definition import NetworkDefinition
from src.network.population import NeuronPopulationSpec


def test_current_attribution_preserves_source_transmitter_sign():
    snn = NetworkDefinition(
        (
            NeuronPopulationSpec("EXCITATORY", 1),
            NeuronPopulationSpec("INHIBITORY", 1, output="inhibitory"),
            NeuronPopulationSpec("TARGET", 1),
        ),
        (
            ConnectionSpec("EXCITATORY", "TARGET", FanOutSpec(1), StrengthSpec(0.3)),
            ConnectionSpec("INHIBITORY", "TARGET", FanOutSpec(1), StrengthSpec(0.2)),
        ),
    ).compile(np.random.default_rng(1))
    snn.neurons.voltage[:] = 3.0
    spikes = snn.step()

    currents = incoming_projection_currents(snn, spikes, snn.layout.population("TARGET"))

    assert currents == {"EXCITATORY_to_TARGET": 0.3, "INHIBITORY_to_TARGET": -0.2}
