import numpy as np
import pytest

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.cognition.readout import Action
from continual_agent.language.spiking_decoder import SpikingCharacterDecoder
from continual_agent.simulation.population_layout import Population, PopulationLayout


def test_spiking_decoder_populations_are_in_main_network() -> None:
    agent = ConversationAgent()

    assert agent.layout.slice(Population.OUTPUT_CHAR).start < agent.network.neurons.count
    assert agent.language.neuron_count == (
        len(agent.language.tokens) * agent.config.neurons_per_token
    )


def test_local_teacher_alignment_changes_token_synapses() -> None:
    agent = ConversationAgent()
    before = agent.network.synapses.weight[agent.token_input_edge_indices].copy()

    agent.train_response_events(
        Action.ANSWER,
        agent.language.target_tokens("hello"),
    )

    after = agent.network.synapses.weight[agent.token_input_edge_indices]
    assert not np.array_equal(before, after)


def test_event_teacher_updates_existing_recurrent_transition_edges() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    edges = agent.recurrent_event_edge_indices
    target = agent.layout.subgroup(Population.OUTPUT_CHAR, "a")
    selected = edges[
        np.isin(agent.network.synapses.target[edges], np.arange(target.start, target.stop))
    ]
    before = agent.network.synapses.weight[selected].copy()

    agent.train_response_events(Action.ANSWER, ("m", "a"))

    assert selected.size
    assert np.any(agent.network.synapses.weight[selected] != before)


def test_repeated_event_has_a_recurrent_character_path_without_input_feedback() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    currents: list[np.ndarray] = []
    original_step = agent.network.step

    def step(current: np.ndarray) -> np.ndarray:
        currents.append(current.copy())
        return original_step(current)

    agent.network.step = step
    m = agent.layout.subgroup(Population.OUTPUT_CHAR, "m")
    m_edges = agent.recurrent_event_edge_indices[
        np.isin(
            agent.network.synapses.target[agent.recurrent_event_edge_indices],
            np.arange(m.start, m.stop),
        )
    ]
    transition_edges = m_edges[
        np.isin(agent.network.synapses.source[m_edges], np.arange(m.start, m.stop))
    ]
    before = agent.network.synapses.weight[m_edges].copy()
    transition_before = agent.network.synapses.weight[transition_edges].copy()
    agent.train_response_events(Action.ANSWER, ("m", "m", "<EOS>"))

    assert currents
    assert np.any(currents[0][: agent.config.input_features])
    assert all(not np.any(current[: agent.config.input_features]) for current in currents[1:])
    assert m_edges.size
    assert np.any(agent.network.synapses.weight[m_edges] != before)
    assert transition_edges.size
    assert np.any(agent.network.synapses.weight[transition_edges] != transition_before)


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
    original_step = agent.network.step
    currents: list[np.ndarray] = []
    ticks: list[int] = []
    states: list[dict[str, np.ndarray | int]] = []

    def step(current: np.ndarray) -> np.ndarray:
        currents.append(current.copy())
        ticks.append(agent.network.tick)
        if agent.network.tick in (0, 3, 6):
            states.append(agent.network.state_snapshot())
        return original_step(current)

    agent.network.step = step
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
    first_ticks = persistent.network.tick
    persistent.generate_response(Action.ANSWER, max_tokens=1)
    assert persistent.network.tick > first_ticks

    reset = ConversationAgent(
        AgentConfig(input_features=16, seed=4, persistent_working_memory=False)
    )
    reset.generate_response(Action.ANSWER, max_tokens=1)
    first_ticks = reset.network.tick
    reset.generate_response(Action.ANSWER, max_tokens=1)
    assert reset.network.tick == first_ticks


def test_token_projection_requires_matching_named_population() -> None:
    decoder = SpikingCharacterDecoder(alphabet=("a",))

    with pytest.raises(ValueError, match="output_char"):
        decoder.token_projection_indices(PopulationLayout(char_count=decoder.neuron_count + 1))
