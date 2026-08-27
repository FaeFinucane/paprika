import numpy as np
import pytest

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.agent.session import InputSignal
from continual_agent.cognition.readout import Action
from continual_agent.language.spiking_decoder import SpikingCharacterDecoder
from continual_agent.simulation.population_layout import Population, PopulationLayout


def test_spiking_decoder_populations_are_in_main_network() -> None:
    agent = ConversationAgent()

    assert (
        agent.runtime.layout.slice(Population.OUTPUT_CHAR).start
        < agent.runtime.network.neurons.count
    )
    assert agent.language.neuron_count == (
        len(agent.language.tokens) * agent.config.neurons_per_token
    )


def test_local_teacher_alignment_changes_token_synapses() -> None:
    agent = ConversationAgent()
    before = agent.runtime.network.synapses.weight[agent.runtime.token_input_edge_indices].copy()

    agent.train_response_events(
        Action.ANSWER,
        agent.language.target_tokens("hello"),
    )

    after = agent.runtime.network.synapses.weight[agent.runtime.token_input_edge_indices]
    assert not np.array_equal(before, after)


def test_response_event_training_restores_idle_lifecycle() -> None:
    agent = ConversationAgent()
    agent.train_response_events(Action.ANSWER, agent.language.target_tokens("hi"))

    assert agent.runtime.response_session.state.value == "idle"
    assert agent.runtime.output_readout.events == []


def test_supervised_teacher_updates_existing_hidden_output_edges() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    edges = agent.runtime.hidden_output_edge_indices
    target = agent.runtime.layout.subgroup(Population.OUTPUT_CHAR, "a")
    selected = edges[
        np.isin(agent.runtime.network.synapses.target[edges], np.arange(target.start, target.stop))
    ]
    before = agent.runtime.network.synapses.weight[selected].copy()

    frame = np.zeros(agent.config.input_features)
    frame[2] = 5.0
    agent.runtime.train_input_events((InputSignal.INPUT_BEGIN, (frame, "a"), InputSignal.INPUT_END))

    assert selected.size
    assert np.any(agent.runtime.network.synapses.weight[selected] != before)


def test_repeated_event_has_no_output_feedback() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    currents: list[np.ndarray] = []
    original_step = agent.runtime.network.step

    def step(current: np.ndarray) -> np.ndarray:
        currents.append(current.copy())
        return original_step(current)

    agent.runtime.network.step = step
    agent.train_response_events(Action.ANSWER, ("m", "m", "<EOS>"))

    assert currents
    assert np.any(currents[0][: agent.config.input_features])
    assert all(not np.any(current[: agent.config.input_features]) for current in currents[1:])
    output = agent.runtime.layout.slice(Population.OUTPUT_CHAR)
    assert not np.any(
        np.isin(agent.runtime.network.synapses.source, np.arange(output.start, output.stop))
    )


def test_response_teacher_respects_ablation() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    edges = agent.runtime.hidden_output_edge_indices.copy()
    agent.runtime.ablate_edges(edges)

    agent.train_response_events(Action.ANSWER, ("m", "<EOS>"))

    np.testing.assert_array_equal(agent.runtime.network.synapses.weight[edges], 0.0)


def test_spiking_response_has_bounded_output_and_eos_control() -> None:
    agent = ConversationAgent()
    response = agent.generate_response(Action.ANSWER, max_tokens=12)

    assert len(response.tokens) <= 12
    assert response.ticks <= 12
    assert all(token in agent.language.alphabet for token in response.tokens)


def test_recurrent_state_spans_output_events_without_external_token_input() -> None:
    agent = ConversationAgent(
        AgentConfig(input_features=16, seed=4, persistent_working_memory=False)
    )
    original_step = agent.runtime.network.step
    currents: list[np.ndarray] = []
    ticks: list[int] = []
    states: list[dict[str, np.ndarray | int]] = []

    def step(current: np.ndarray) -> np.ndarray:
        currents.append(current.copy())
        ticks.append(agent.runtime.network.tick)
        if agent.runtime.network.tick in (0, 3, 6):
            states.append(agent.runtime.network.state_snapshot())
        return original_step(current)

    agent.runtime.network.step = step
    response = agent.generate_response(Action.ANSWER, max_tokens=3)

    assert response.ticks <= 3
    assert ticks == list(range(len(ticks)))
    assert states and states[0]["tick"] == 0
    # Output events share recurrent state; decoded characters are never fed
    # back through the external input population.
    assert np.any(currents[0][: agent.config.input_features])
    for start in range(3, len(currents), 3):
        np.testing.assert_array_equal(
            currents[start][: agent.config.input_features],
            np.zeros(agent.config.input_features),
        )


def test_recurrent_state_boundary_follows_session_policy() -> None:
    persistent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    persistent.generate_response(Action.ANSWER, max_tokens=1)
    first_ticks = persistent.runtime.network.tick
    persistent.generate_response(Action.ANSWER, max_tokens=1)
    assert persistent.runtime.network.tick > first_ticks

    reset = ConversationAgent(
        AgentConfig(input_features=16, seed=4, persistent_working_memory=False)
    )
    reset.generate_response(Action.ANSWER, max_tokens=1)
    first_ticks = reset.runtime.network.tick
    reset.generate_response(Action.ANSWER, max_tokens=1)
    assert reset.runtime.network.tick == first_ticks


def test_token_projection_requires_matching_named_population() -> None:
    decoder = SpikingCharacterDecoder(alphabet=("a",))

    with pytest.raises(ValueError, match="output_char"):
        decoder.token_projection_indices(PopulationLayout(char_count=decoder.neuron_count + 1))
