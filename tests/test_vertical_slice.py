import numpy as np
import pytest

from continual_agent.agent.session import (
    InputSignal,
    ResponseSession,
    SessionSnapshot,
    SessionStateError,
)
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.population_layout import Population
from continual_agent.simulation.synapses import SparseSynapses
from tests.helpers import network_config


def test_runtime_boundaries_drive_distinct_input_channels() -> None:
    runtime = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=2,
        )
    )
    runtime.start_session()
    begin = runtime.step_input_signal(InputSignal.INPUT_BEGIN)
    assert begin[0]
    assert runtime.network.neurons.voltage[0] == 0.0
    end = runtime.step_input_signal(InputSignal.INPUT_END)
    assert end[1]
    assert not np.array_equal(begin, end)


def test_background_drive_is_seeded_and_vectorized() -> None:
    first = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=21,
            background_rate=0.5,
            background_current=0.4,
        )
    )
    second = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=21,
            background_rate=0.5,
            background_current=0.4,
        )
    )
    current = np.zeros(first.network.neurons.count)
    first_spikes = [first.step(current) for _ in range(8)]
    second_spikes = [second.step(current) for _ in range(8)]
    np.testing.assert_array_equal(first_spikes, second_spikes)
    assert first.diagnostics["firing_rate"] >= 0.0


def test_normal_output_pathways_are_not_zeroed() -> None:
    runtime = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=22,
        )
    )
    output = runtime.layout.slice(Population.OUTPUT_CHAR)
    incoming = (runtime.network.synapses.target >= output.start) & (
        runtime.network.synapses.target < output.stop
    )
    assert np.all(runtime.network.synapses.weight[incoming] != 0.0)


def test_boundary_current_is_neural_and_not_a_semantic_feature() -> None:
    runtime = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=9,
        )
    )
    runtime.start_session()
    boundary = runtime.boundary_current(InputSignal.INPUT_END)
    assert boundary[1]
    assert not boundary[0]
    assert runtime.network.tick == 0
    emitted = runtime.step_input_signal(InputSignal.INPUT_BEGIN)
    assert runtime.network.tick == 1
    assert np.any(emitted)


def test_session_rejects_overlapping_input_presentation() -> None:
    session = ResponseSession()
    session.begin(SessionSnapshot(np.zeros(1), np.zeros(1, dtype=int), np.zeros(1)))
    session.begin_input()
    try:
        session.begin_input()
    except SessionStateError:
        pass
    else:
        raise AssertionError("overlapping input should be rejected")
    session.end_input()


def test_session_rejects_input_frames_outside_boundaries() -> None:
    session = ResponseSession()
    session.begin(SessionSnapshot(np.zeros(1), np.zeros(1, dtype=int), np.zeros(1)))

    with pytest.raises(SessionStateError, match="outside"):
        session.accept_input_frame()

    session.begin_input()
    session.accept_input_frame()
    session.end_input()


def test_reward_modulated_stdp_changes_only_selected_targets() -> None:
    synapses = SparseSynapses(
        source=np.array([0, 0]),
        target=np.array([1, 2]),
        weight=np.array([0.2, 0.2]),
        neuron_count=3,
    )
    plasticity = RewardModulatedSTDP(synapses, learning_rate=0.1)
    plasticity.observe(np.array([True, False, False]))
    plasticity.observe(np.array([False, True, False]))
    before = synapses.weight.copy()

    plasticity.reinforce(1.0, target_neurons=np.array([1]))

    assert synapses.weight[0] > before[0]
    assert synapses.weight[1] == before[1]
