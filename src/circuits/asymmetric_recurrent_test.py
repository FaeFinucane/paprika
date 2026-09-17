"""Viability envelope for the asymmetric recurrent temporal circuit."""

import numpy as np
import pytest

from .vta_dopamine import build_dopamine_circuit

pytestmark = pytest.mark.viability


def test_sparse_state_seeding_recruits_a_bounded_asymmetric_recurrent_circuit():
    """The circuit produces a bounded cue-evoked temporal trajectory."""
    for seed in range(4):
        circuit = build_dopamine_circuit(seed, enable_dopamine_learning=False)
        _tick(circuit, 240)
        cue = np.zeros(circuit.populations["CUE"].count)
        cue[: cue.size // 2] = 1.0
        circuit.present_cue(cue, 5)
        state_rates = []
        temporal_rates = []
        inhibitory_rates = []
        for _ in range(60):
            spikes = circuit.tick()
            state_rates.append(
                float(
                    np.mean(
                        spikes.population(circuit.populations[circuit.inferred_state.excitatory])
                    )
                )
            )
            temporal_rates.append(
                float(np.mean(spikes.population(circuit.populations[circuit.temporal.excitatory])))
            )
            inhibitory_rates.append(
                float(np.mean(spikes.population(circuit.populations["VTA_INHIB"])))
            )

        assert max(state_rates) >= 0.25
        assert max(temporal_rates) >= 0.20
        assert max(temporal_rates[8:16]) >= 0.05
        assert max(inhibitory_rates) >= 1 / circuit.populations["VTA_INHIB"].count
        assert max(temporal_rates[30:]) <= 0.15
        synapses = circuit.session.snn.synapses
        recurrent = synapses.projection_mask("TEMPORAL_E_recurrent")
        count = circuit.populations[circuit.temporal.excitatory].count
        offsets = (synapses.target[recurrent] - synapses.source[recurrent]) % count
        assert np.all((1 <= offsets) & (offsets <= 8))


def _tick(circuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()
