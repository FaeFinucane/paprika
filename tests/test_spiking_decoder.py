import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.readout import Action


def test_spiking_decoder_populations_are_in_main_network() -> None:
    agent = ConversationAgent()

    assert agent.token_start < agent.network.neurons.count
    assert agent.language.neuron_count == (
        len(agent.language.tokens) * agent.config.neurons_per_token
    )


def test_local_teacher_alignment_changes_token_synapses() -> None:
    agent = ConversationAgent()
    before = agent.network.synapses.weight[agent.token_input_edge_indices].copy()

    agent.train_response_text(Action.ANSWER, "hello")

    after = agent.network.synapses.weight[agent.token_input_edge_indices]
    assert not np.array_equal(before, after)


def test_spiking_response_has_bounded_output_and_eos_control() -> None:
    agent = ConversationAgent()
    response = agent.generate_response(Action.ANSWER, max_tokens=12)

    assert len(response.tokens) <= 12
    assert response.ticks <= 12
    assert all(token in agent.language.alphabet for token in response.tokens)
