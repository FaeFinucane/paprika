"""Viability envelope for the asymmetric recurrent temporal circuit."""

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.interaction.rates import RatePatternInput
from src.network.connectivity import FanInSpec, StrengthSpec

from .asymmetric_recurrent import (
    AsymmetricRecurrentSpec,
    TargetWindowFanOutSpec,
    add_asymmetric_recurrent_circuit,
)
from .attractor import AttractorSpec, add_attractor

pytestmark = pytest.mark.viability


def test_sparse_state_seeding_recruits_a_bounded_asymmetric_recurrent_circuit():
    """The circuit produces a bounded cue-evoked temporal trajectory."""
    for seed in range(4):
        builder = NetworkBuilder()
        cue_handle = builder.add_feature_population("CUE", ("CUE",), width=16)
        state = add_attractor(builder, AttractorSpec())
        builder.connect(
            cue_handle,
            state.input,
            FanInSpec(6),
            StrengthSpec(0.25, 0.02, maximum=0.8),
            "dopamine_stdp",
            True,
        )
        temporal_spec = AsymmetricRecurrentSpec()
        temporal = add_asymmetric_recurrent_circuit(builder, temporal_spec)
        builder.connect(
            state.output,
            temporal.input,
            TargetWindowFanOutSpec(3, target_start=0, target_count=8),
            StrengthSpec(0.30, 0.02, maximum=0.8),
            "dopamine_stdp",
            True,
        )
        session = builder.compile(seed)
        populations = {
            population.spec.name: population for population in session.snn.layout.populations
        }
        cue_input = RatePatternInput(populations[cue_handle.name], np.random.default_rng(seed))
        session.add(cue_input)
        _tick(session, 240)
        cue = np.zeros(populations[cue_handle.name].count)
        cue[: cue.size // 2] = 1.0
        cue_input.write(cue, 5)
        state_rates = []
        temporal_rates = []
        for _ in range(60):
            spikes = session.tick()
            state_rates.append(float(np.mean(spikes.population(populations[state.excitatory]))))
            temporal_rates.append(
                float(np.mean(spikes.population(populations[temporal.excitatory])))
            )

        assert max(state_rates) >= 0.25
        assert max(temporal_rates) >= 0.20
        assert max(temporal_rates[8:16]) >= 0.05
        assert max(temporal_rates[30:]) <= 0.15
        synapses = session.snn.synapses
        recurrent = synapses.projection_mask("TEMPORAL_E_recurrent")
        count = populations[temporal.excitatory].count
        offsets = (synapses.target[recurrent] - synapses.source[recurrent]) % count
        assert np.all((1 <= offsets) & (offsets <= 8))


def _tick(session, ticks: int) -> None:
    for _ in range(ticks):
        session.tick()
