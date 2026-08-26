import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.simulation.population_layout import Population


def test_affect_to_action_pathway_is_causally_ablatable() -> None:
    agent = ConversationAgent()
    edges = agent.affect_action_edge_indices
    original = agent.network.synapses.weight[edges].copy()

    # Make the pathway deliberately visible for this causal test. Production
    # weights remain learned and are not hard-coded this way.
    agent.network.synapses.weight[edges] = 0.8
    agent.respond("Can you help me choose?")
    with_pathway = np.asarray(agent.network.spike_history)
    output = agent.layout.slice(Population.OUTPUT_ACTION)
    with_output_voltage = agent.network.neurons.voltage[output.start:].copy()

    agent.network.synapses.weight[edges] = 0.0
    agent.respond("Can you help me choose?")
    without_pathway = np.asarray(agent.network.spike_history)
    without_output_voltage = agent.network.neurons.voltage[output.start:].copy()

    agent.network.synapses.weight[edges] = original
    assert with_pathway[:, output.start:].sum() > 0
    assert without_pathway[:, output.start:].sum() > 0
    assert not np.array_equal(with_output_voltage, without_output_voltage)


def test_affect_population_is_part_of_main_network_state() -> None:
    agent = ConversationAgent()
    affect = agent.layout.slice(Population.AFFECT)
    output = agent.layout.slice(Population.OUTPUT_ACTION)
    assert affect.start < output.start
    assert output.start < agent.network.neurons.count
    assert agent.affect_circuit.neuron_count == (
        len(AffectiveCircuit.signal_names) * agent.config.neurons_per_affect
    )

    agent.respond("Please explain this clearly.")
    affect_activity = np.asarray(agent.network.spike_history)[:, affect.start : output.start]

    assert affect_activity.shape[1] == agent.affect_circuit.neuron_count
    assert affect_activity.sum() > 0
