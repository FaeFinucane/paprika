import numpy as np
import pytest
from src.interaction.plasticity import DopamineSTDP
from src.network.adjustments import NetworkAdjustment
from src.network.connectivity import ConnectionSpec, FanOutSpec, StrengthSpec, build_synapses
from src.network.population import NeuronPopulationSpec, PopulationLayout
from src.network.snn import SNN, Spikes

pytestmark = pytest.mark.unit


def _layout(*specs):
    return PopulationLayout.build(specs)


def test_modulatory_population_cannot_create_ordinary_projection():
    layout = _layout(
        NeuronPopulationSpec("DA", 1, output="modulatory"),
        NeuronPopulationSpec("TARGET", 1),
    )
    spec = ConnectionSpec("DA", "TARGET", FanOutSpec(1), StrengthSpec(0.2))
    with pytest.raises(ValueError, match="modulatory"):
        build_synapses(layout, [spec], np.random.default_rng(1))


def test_inhibitory_sign_is_derived_from_source_and_arrives_on_the_next_tick():
    layout = _layout(
        NeuronPopulationSpec("INHIBITORY", 1, output="inhibitory"),
        NeuronPopulationSpec("TARGET", 1),
    )
    synapses = build_synapses(
        layout,
        [ConnectionSpec("INHIBITORY", "TARGET", FanOutSpec(1), StrengthSpec(0.4))],
        np.random.default_rng(2),
    )
    assert synapses.strength[0] >= 0
    snn = SNN.build(layout, synapses)
    source = layout.population("INHIBITORY")
    snn.step({source: np.array([1.0])})
    target = layout.population("TARGET")
    assert snn.next_current[target.start] < 0
    snn.step()
    assert snn.neurons.voltage[target.start] < 0


def test_neurons_share_fixed_dynamics_and_have_no_implicit_current():
    layout = _layout(NeuronPopulationSpec("A", 1), NeuronPopulationSpec("B", 1))
    snn = SNN.build(layout, build_synapses(layout, [], np.random.default_rng(7)))

    for _ in range(10):
        assert not np.any(snn.step().values)

    assert snn.neurons.dynamics.threshold == 1.0
    assert snn.neurons.dynamics.reset == -0.5
    assert snn.neurons.dynamics.decay == 0.9
    assert snn.neurons.dynamics.refractory_ticks == 0


def test_fixed_connections_are_not_mutable():
    layout = _layout(NeuronPopulationSpec("A", 1), NeuronPopulationSpec("B", 1))
    synapses = build_synapses(
        layout,
        [ConnectionSpec("A", "B", FanOutSpec(1), StrengthSpec(0.2))],
        np.random.default_rng(3),
    )
    with pytest.raises(ValueError, match="fixed"):
        synapses.apply_strength_delta(np.array([0.1]), np.array([True]))
    with pytest.raises(ValueError, match="fixed"):
        ConnectionSpec("A", "B", FanOutSpec(1), StrengthSpec(0.2), scalable=True)


def test_overlapping_strength_proposals_are_combined_before_bounds_are_applied():
    layout = _layout(NeuronPopulationSpec("A", 1), NeuronPopulationSpec("B", 1))
    synapses = build_synapses(
        layout,
        [
            ConnectionSpec(
                "A",
                "B",
                FanOutSpec(1),
                StrengthSpec(0.9),
                learning="homeostatic",
            )
        ],
        np.random.default_rng(8),
    )
    snn = SNN.build(layout, synapses)
    mask = np.array([True])

    snn.commit(
        [
            NetworkAdjustment(np.array([0.2]), mask),
            NetworkAdjustment(np.array([-0.1]), mask),
        ]
    )

    assert synapses.strength[0] == 1.0


class _Signal:
    value = 1.0


def test_dopamine_response_is_taken_from_plastic_target_neuron():
    layout = _layout(
        NeuronPopulationSpec("PRE", 1),
        NeuronPopulationSpec("ALIGNED", 1, dopamine_response="aligned"),
        NeuronPopulationSpec("OPPOSED", 1, dopamine_response="opposed"),
    )
    synapses = build_synapses(
        layout,
        [
            ConnectionSpec("PRE", "ALIGNED", FanOutSpec(1), StrengthSpec(0.2), "dopamine_stdp"),
            ConnectionSpec("PRE", "OPPOSED", FanOutSpec(1), StrengthSpec(0.2), "dopamine_stdp"),
        ],
        np.random.default_rng(4),
    )
    snn = SNN.build(layout, synapses)
    rule = DopamineSTDP(snn, _Signal(), trace_decay=0.5)
    rule.observe(Spikes(np.array([True, False, False]), 1, layout.fingerprint))
    rule.observe(Spikes(np.array([False, True, True]), 2, layout.fingerprint))
    before = synapses.strength.copy()
    snn.commit([rule.propose(snn)])
    assert synapses.strength[0] > before[0]
    assert synapses.strength[1] < before[1]
