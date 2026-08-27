import numpy as np
import pytest

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.agent.network_config import NetworkConfig
from continual_agent.agent.session import (
    ConflictPolicy,
    InputSignal,
    ResponseSession,
    SessionPolicy,
    SessionSnapshot,
    SessionStateError,
)
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.cognition.readout import Action
from continual_agent.cognition.working_memory import WorkingMemory
from continual_agent.encoding.text_encoder import TextEncoder
from continual_agent.environment.scenarios import default_scenarios
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.population_layout import Population
from continual_agent.simulation.synapses import SparseSynapses


def test_runtime_boundaries_drive_distinct_input_channels() -> None:
    runtime = SpikingRuntime(
        NetworkConfig(
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
        NetworkConfig(
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
        NetworkConfig(
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
        NetworkConfig(
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
        NetworkConfig(
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


def test_text_encoding_is_deterministic_and_sparse() -> None:
    encoder = TextEncoder(feature_count=32, ticks_per_token=2)

    first = encoder.encode("Hello there")
    second = encoder.encode("Hello there")

    np.testing.assert_array_equal(first, second)
    assert first.shape[1] == 32
    assert np.count_nonzero(encoder.feature_vector("Hello there")) == 2


def test_input_presentation_has_boundaries_and_preserves_semantic_activity() -> None:
    encoder = TextEncoder(feature_count=16, ticks_per_token=2)
    encoded = encoder.encode("hello")
    events = encoder.present("hello", presentation_speed=2)

    assert events[0].signal is InputSignal.INPUT_BEGIN
    assert events[-1].signal is InputSignal.INPUT_END
    frames = np.asarray([event.frame for event in events[1:-1]])
    np.testing.assert_array_equal(frames[0], encoded[0])
    assert len(frames) == len(encoded) * 2


def test_text_hashing_never_uses_boundary_channels() -> None:
    encoded = TextEncoder(8).encode("boundary")
    assert not np.any(encoded[:, :2])


def test_rate_presentation_holds_each_frame_for_configured_speed() -> None:
    events = TextEncoder(8, ticks_per_token=3).present("x", presentation_speed=2)
    frames = [event for event in events if event.frame is not None]

    assert len(frames) == len(TextEncoder(8, ticks_per_token=3).encode("x")) * 2
    np.testing.assert_array_equal(frames[0].frame, frames[1].frame)


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


def test_agent_applies_boundary_policy_and_captures_execution_snapshot() -> None:
    agent = ConversationAgent(AgentConfig(seed=3, persistent_working_memory=True))
    agent.runtime.response_session.policy = SessionPolicy(
        reset_readout=True,
        reset_neuron_state=True,
        reset_synaptic_activity=True,
        persist_affect=False,
        persist_working_memory=False,
        persist_plasticity_eligibility=False,
    )
    agent.runtime.network.neurons.voltage[0] = 0.7
    agent.runtime.network.reset_synaptic_activity()
    agent.working_memory.state[0] = 0.8
    agent.affect.valence = 0.9
    agent.runtime.plasticity.eligibility[0] = 0.6

    agent._start_response_session()
    saved = agent.runtime.response_session.snapshot
    assert saved is not None
    assert saved.neuron_voltage[0] == 0.0
    assert saved.working_memory is not None
    assert isinstance(saved.working_memory, WorkingMemory)
    assert saved.working_memory.state[0] == 0.0
    assert agent.affect.valence == 0.0
    assert agent.runtime.plasticity.eligibility[0] == 0.0
    assert agent.last_execution_snapshot is not None
    agent.runtime.response_session.begin_response()
    agent.runtime.response_session.abort()


def test_missing_response_eos_aborts_the_agent_session() -> None:
    agent = ConversationAgent(AgentConfig(seed=4))

    response = agent.generate_response(Action.ANSWER, max_tokens=0)

    assert not response.stopped_on_eos
    assert agent.runtime.response_session.state.value == "exhausted"
    assert agent.last_execution_snapshot is not None


def test_response_exception_cleans_up_session() -> None:
    agent = ConversationAgent(AgentConfig(seed=4))
    original = agent.runtime.step

    def fail(current: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")

    agent.runtime.step = fail
    with pytest.raises(RuntimeError):
        agent.respond("hello")
    assert agent.runtime.response_session.state.value == "idle"
    agent.runtime.step = original


def test_agent_input_presentation_closes_before_response() -> None:
    agent = ConversationAgent(AgentConfig(seed=5, max_thinking_ticks=1))

    agent.respond("hello")

    assert agent.runtime.response_session.input_active is False
    assert agent.runtime.response_session.state.value in {"complete", "idle"}


def test_zero_token_limit_is_explicit_exhaustion_without_network_ticks() -> None:
    agent = ConversationAgent(AgentConfig(seed=6))

    response = agent.generate_response(Action.ANSWER, max_tokens=0)

    assert response.tokens == ()
    assert response.ticks == 0
    assert not response.stopped_on_eos
    assert agent.runtime.network.tick == 0
    assert agent.runtime.response_session.state.value == "exhausted"


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


def test_isolated_sessions_do_not_share_execution_state_and_merge_explicitly() -> None:
    agent = ConversationAgent(AgentConfig(seed=17, max_thinking_ticks=1))
    before_weights = agent.runtime.network.synapses.weight.copy()
    before_voltage = agent.runtime.network.neurons.voltage.copy()

    first, first_execution = agent.train_response_isolated("hello", Action.ANSWER, merge=False)
    second, second_execution = agent.train_response_isolated("hello", Action.REVISE, merge=False)

    assert first.action in tuple(Action)
    assert second.action in tuple(Action)
    np.testing.assert_array_equal(agent.runtime.network.synapses.weight, before_weights)
    np.testing.assert_array_equal(agent.runtime.network.neurons.voltage, before_voltage)
    assert first_execution.updates
    assert second_execution.updates
    assert first_execution is not second_execution

    first_execution.merge_into(agent.runtime.network.synapses.weight, ConflictPolicy.REJECT)
    after_first = agent.runtime.network.synapses.weight.copy()
    second_execution.merge_into(agent.runtime.network.synapses.weight, ConflictPolicy.SUM)
    assert not np.array_equal(agent.runtime.network.synapses.weight, before_weights)
    assert not np.array_equal(agent.runtime.network.synapses.weight, after_first)


def test_isolated_response_does_not_merge_non_weight_state() -> None:
    agent = ConversationAgent(AgentConfig(seed=18, max_thinking_ticks=1))
    before_tick = agent.runtime.network.tick
    before_snapshot = agent.debug_snapshot()

    decision, execution = agent.respond_isolated("hello", merge=True)

    assert decision.action in tuple(Action)
    assert agent.runtime.network.tick == before_tick
    assert agent.debug_snapshot() == before_snapshot
    assert execution.updates == ()


def test_isolated_execution_does_not_merge_by_default() -> None:
    agent = ConversationAgent(AgentConfig(seed=19, max_thinking_ticks=1))
    before = agent.runtime.network.synapses.weight.copy()

    _, execution = agent.train_response_isolated("hello", Action.ANSWER)

    assert execution.updates
    np.testing.assert_array_equal(agent.runtime.network.synapses.weight, before)


def test_isolated_execution_copies_output_arbitration_policy() -> None:
    agent = ConversationAgent(AgentConfig(seed=20))
    policy = agent.runtime.output_readout.arbitration

    def mutate_policy(isolated: ConversationAgent) -> None:
        assert isolated.runtime.output_readout.arbitration is not policy
        isolated.runtime.output_readout.arbitration.action_priority = 99

    agent.execute_isolated(mutate_policy)

    assert policy.action_priority == 1


def test_scripted_curriculum_learns_all_initial_intents() -> None:
    agent = ConversationAgent(AgentConfig(seed=0))
    scenarios = default_scenarios()

    for _ in range(30):
        for scenario in scenarios:
            agent.train_response(scenario.messages[0], scenario.expected)

    results = [agent.respond(s.messages[0]).action for s in scenarios]

    assert len(results) == len(scenarios)
