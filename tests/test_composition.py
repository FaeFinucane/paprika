from copy import deepcopy

import numpy as np

from continual_agent.agent.drives import ArrayDrive, DriveAggregator
from continual_agent.agent.input_runner import InputRunner
from continual_agent.agent.network_config import NetworkConfig
from continual_agent.agent.plugins import NetworkContext
from continual_agent.agent.runtime_session import RuntimeSession
from continual_agent.agent.session import InputSignal
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.simulation import LIFNeurons, NetworkCore, SparseSynapses
from continual_agent.simulation.population_layout import Population
from continual_agent.simulation.weight_initialization import (
    PopulationProjectionSeed,
    WeightInitializationConfig,
)


def test_network_core_preserves_one_tick_synaptic_delay_and_snapshots() -> None:
    core = NetworkCore(
        LIFNeurons(2, tau_membrane=1.0, threshold=1.0, refractory_ticks=0),
        SparseSynapses(np.array([0]), np.array([1]), np.array([1.0]), 2),
    )
    current = np.array([2.0, 0.0])
    first = core.step(current)
    snapshot = core.state_snapshot()
    second = core.step(np.zeros(2))

    np.testing.assert_array_equal(first, [True, False])
    np.testing.assert_array_equal(second, [False, True])
    assert snapshot["tick"] == 1
    np.testing.assert_allclose(snapshot["pending_current"], [0.0, 1.0])


def test_drive_aggregator_reuses_current_buffer() -> None:
    aggregator = DriveAggregator(2)
    external = ArrayDrive(2)
    external.current[:] = [1.0, 2.0]
    aggregator.add(external)

    first = aggregator.collect()
    external.current[:] = [3.0, 4.0]
    second = aggregator.collect()

    assert first is second
    np.testing.assert_allclose(second, [3.0, 4.0])


def test_network_plugin_context_can_be_scheduled_without_neuron_loops() -> None:
    seen: list[NetworkContext] = []

    class Plugin:
        interval = 2

        def after_step(self, context: NetworkContext) -> None:
            seen.append(context)

    core = NetworkCore.random(4, seed=4, connection_probability=0.0)
    plugin = Plugin()
    for _ in range(4):
        spikes = core.step(np.zeros(4))
        if core.tick % plugin.interval == 0:
            plugin.after_step(NetworkContext(core.tick, spikes, core.neurons.voltage))

    assert [context.tick for context in seen] == [2, 4]


def test_runtime_composes_owned_config_session_and_runner_components() -> None:
    runtime = SpikingRuntime(
        NetworkConfig(
            input_features=4,
            hidden_neurons=3,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=8,
        )
    )

    assert isinstance(runtime.session, RuntimeSession)
    assert isinstance(runtime.trainer, InputRunner)
    assert not hasattr(runtime, "neurons")
    assert not hasattr(runtime, "synapses")
    assert not hasattr(runtime, "background_rate")
    assert not hasattr(runtime, "background_current")
    assert runtime.background_drive.rate == 0.0
    assert runtime.background_drive.current == 0.05
    config = NetworkConfig(4, 3, ("<EOS>", "A"), 1)
    before = deepcopy(config)
    config.build()
    assert config == before
    assert not hasattr(runtime, "_diagnostic_ticks")
    assert runtime.metrics.ticks == 0


def test_network_config_rejects_nonfinite_and_invalid_values() -> None:
    import pytest

    with pytest.raises(ValueError):
        NetworkConfig(4, 3, ("<EOS>",), 1, connection_probability=np.nan)
    with pytest.raises(ValueError):
        NetworkConfig(4, 0, ("<EOS>",), 1)


def test_isolated_clone_preserves_custom_plugins() -> None:
    runtime = SpikingRuntime(NetworkConfig(4, 3, ("<EOS>",), 1))

    class Plugin:
        interval = 1

        def after_step(self, context: NetworkContext) -> None:
            pass

    plugin = Plugin()
    runtime.plugins.append(plugin)
    clone = runtime.session.clone()
    assert any(type(item) is Plugin for item in clone.plugins)


def test_hidden_recurrence_is_exposed() -> None:
    runtime = SpikingRuntime(
        NetworkConfig(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            connection_probability=1.0,
            seed=8,
        )
    )
    recurrence = runtime.hidden_recurrent_edge_indices
    hidden = runtime.layout.slice(Population.HIDDEN)
    assert recurrence.size > 0
    assert np.all(
        np.isin(runtime.network.synapses.source[recurrence], np.arange(hidden.start, hidden.stop))
    )
    assert np.all(
        np.isin(runtime.network.synapses.target[recurrence], np.arange(hidden.start, hidden.stop))
    )


def test_population_projection_seed_is_reproducible_bounded_and_dense() -> None:
    weights = WeightInitializationConfig(
        population_projections=(
            PopulationProjectionSeed(Population.HIDDEN, Population.HIDDEN, 0.4),
        )
    )
    config = NetworkConfig(
        4, 4, ("<EOS>",), 1, connection_probability=0.0, seed=21, weight_initialization=weights
    )
    first = SpikingRuntime(config)
    second = SpikingRuntime(config)
    first_edges = first.hidden_recurrent_edge_indices
    second_edges = second.hidden_recurrent_edge_indices
    np.testing.assert_array_equal(
        first.network.synapses.source[first_edges], second.network.synapses.source[second_edges]
    )
    np.testing.assert_array_equal(
        first.network.synapses.target[first_edges], second.network.synapses.target[second_edges]
    )
    np.testing.assert_array_equal(
        first.network.synapses.weight[first_edges], second.network.synapses.weight[second_edges]
    )
    assert first_edges.size == 12
    assert np.all(
        (first.network.synapses.weight[first_edges] >= -1.0)
        & (first.network.synapses.weight[first_edges] <= 1.0)
    )


def test_delayed_copy_baseline_depends_on_hidden_retention() -> None:
    config = NetworkConfig(
        input_features=4,
        hidden_neurons=8,
        output_tokens=("<EOS>", "A"),
        neurons_per_token=1,
        connection_probability=1.0,
        seed=8,
    )
    frame = np.array([0.0, 0.0, 5.0, 0.0])
    ordinary = SpikingRuntime(config)
    retention = SpikingRuntime(config)
    events = (InputSignal.INPUT_BEGIN, (frame, "A"), InputSignal.INPUT_END)
    ordinary.train_input_events(events)
    retention.train_delayed_copy_baseline(events)
    indices = retention.hidden_recurrent_edge_indices
    group = retention.hidden_feature_groups[2] - retention.input_features
    source = retention.network.synapses.source[indices] - retention.input_features
    target = retention.network.synapses.target[indices] - retention.input_features
    active_group_edges = indices[np.isin(source, group) & np.isin(target, group)]
    other_edges = indices[~(np.isin(source, group) & np.isin(target, group))]
    assert active_group_edges.size > 0
    assert np.any(
        retention.network.synapses.weight[active_group_edges]
        != ordinary.network.synapses.weight[active_group_edges]
    )
    np.testing.assert_array_equal(
        retention.network.synapses.weight[other_edges],
        ordinary.network.synapses.weight[other_edges],
    )
