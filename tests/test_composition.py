import numpy as np

from continual_agent.agent.drives import ArrayDrive, DriveAggregator
from continual_agent.agent.input_runner import InputRunner
from continual_agent.agent.network_factory import NetworkFactory
from continual_agent.agent.plugins import NetworkContext
from continual_agent.agent.runtime_session import RuntimeSession
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.simulation import LIFNeurons, NetworkCore, SparseSynapses


def test_network_core_preserves_one_tick_synaptic_delay_and_snapshots() -> None:
    core = NetworkCore(
        LIFNeurons(2, tau_membrane=1.0, threshold=1.0, refractory_ticks=0),
        SparseSynapses(np.array([0]), np.array([1]), np.array([2.0]), 2),
    )
    current = np.array([2.0, 0.0])
    first = core.step(current)
    snapshot = core.state_snapshot()
    second = core.step(np.zeros(2))

    np.testing.assert_array_equal(first, [True, False])
    np.testing.assert_array_equal(second, [False, True])
    assert snapshot["tick"] == 1
    np.testing.assert_allclose(snapshot["pending_current"], [0.0, 2.0])


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


def test_runtime_composes_owned_factory_session_and_runner_components() -> None:
    runtime = SpikingRuntime(
        input_features=4,
        hidden_neurons=3,
        output_tokens=("<EOS>", "A"),
        neurons_per_token=1,
        seed=8,
    )

    assert isinstance(runtime.session, RuntimeSession)
    assert isinstance(runtime.trainer, InputRunner)
    assert not hasattr(runtime, "neurons")
    assert not hasattr(runtime, "synapses")
    assert not hasattr(runtime, "background_rate")
    assert not hasattr(runtime, "background_current")
    assert runtime.background_drive.rate == 0.0
    assert runtime.background_drive.current == 0.05
    assert isinstance(NetworkFactory(input_features=4), NetworkFactory)
    assert not hasattr(runtime, "_diagnostic_ticks")
    assert runtime.metrics.ticks == 0
