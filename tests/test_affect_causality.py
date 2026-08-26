import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.affect_circuit import AffectiveCircuit


def test_affect_to_action_pathway_is_causally_ablatable() -> None:
    agent = ConversationAgent()
    edges = agent.affect_action_edge_indices
    original = agent.network.synapses.weight[edges].copy()

    # Make the pathway deliberately visible for this causal test. Production
    # weights remain learned and are not hard-coded this way.
    agent.network.synapses.weight[edges] = 0.8
    agent.respond("Can you help me choose?")
    with_pathway = np.asarray(agent.network.spike_history)
    with_output_spikes = with_pathway[:, agent.output_start:].sum()

    agent.network.synapses.weight[edges] = 0.0
    agent.respond("Can you help me choose?")
    without_pathway = np.asarray(agent.network.spike_history)
    without_output_spikes = without_pathway[:, agent.output_start:].sum()

    agent.network.synapses.weight[edges] = original
    assert with_output_spikes != without_output_spikes


def test_affect_population_is_part_of_main_network_state() -> None:
    agent = ConversationAgent()
    assert agent.affect_start < agent.output_start
    assert agent.output_start < agent.network.neurons.count
    assert agent.affect_circuit.neuron_count == (
        len(AffectiveCircuit.signal_names) * agent.config.neurons_per_affect
    )

    agent.respond("Please explain this clearly.")
    affect_activity = np.asarray(agent.network.spike_history)[:, agent.affect_start : agent.output_start]

    assert affect_activity.shape[1] == agent.affect_circuit.neuron_count
    assert affect_activity.sum() > 0
