import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.simulation.population_layout import Population


def test_affect_to_action_pathway_is_causally_ablatable() -> None:
    agent = ConversationAgent()
    edges = agent.runtime.affect_action_edge_indices
    original = agent.runtime.network.synapses.weight[edges].copy()

    # Make the pathway deliberately visible for this causal test. Production
    # weights remain learned and are not hard-coded this way.
    agent.runtime.network.synapses.weight[edges] = 0.8
    with_pathway = _respond_with_trace(agent, "Can you help me choose?")
    output = agent.runtime.layout.slice(Population.OUTPUT_ACTION)
    with_output_voltage = agent.runtime.network.neurons.voltage[output.start :].copy()

    agent.runtime.network.synapses.weight[edges] = 0.0
    _respond_with_trace(agent, "Can you help me choose?")
    without_output_voltage = agent.runtime.network.neurons.voltage[output.start :].copy()

    agent.runtime.network.synapses.weight[edges] = original
    assert with_pathway[:, output.start :].sum() > 0
    assert not np.array_equal(with_output_voltage, without_output_voltage)


def test_affect_population_is_part_of_main_network_state() -> None:
    agent = ConversationAgent()
    affect = agent.runtime.layout.slice(Population.AFFECT)
    output = agent.runtime.layout.slice(Population.OUTPUT_ACTION)
    assert affect.start < output.start
    assert output.start < agent.runtime.network.neurons.count
    assert agent.affect_circuit.neuron_count == (
        len(AffectiveCircuit.signal_names) * agent.config.neurons_per_affect
    )

    affect_activity = _respond_with_trace(agent, "Please explain this clearly")[
        :, affect.start : output.start
    ]

    assert affect_activity.shape[1] == agent.affect_circuit.neuron_count
    assert affect_activity.sum() > 0


def _respond_with_trace(agent: ConversationAgent, text: str) -> np.ndarray:
    """Collect only the short trace needed by this causal assertion."""
    trace: list[np.ndarray] = []
    step = agent.runtime.network.step

    def traced_step(external_current: np.ndarray | None = None) -> np.ndarray:
        emitted = step(external_current)
        trace.append(emitted.copy())
        return emitted

    agent.runtime.network.step = traced_step
    try:
        agent.respond(text)
    finally:
        agent.runtime.network.step = step
    return np.asarray(trace)
