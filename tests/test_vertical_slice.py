import numpy as np

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.cognition.readout import Action
from continual_agent.encoding.text_encoder import TextEncoder
from continual_agent.environment.scenarios import default_scenarios
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.synapses import SparseSynapses


def test_text_encoding_is_deterministic_and_sparse() -> None:
    encoder = TextEncoder(feature_count=32, ticks_per_token=2)

    first = encoder.encode("Hello there")
    second = encoder.encode("Hello there")

    np.testing.assert_array_equal(first, second)
    assert first.shape[1] == 32
    assert np.count_nonzero(encoder.feature_vector("Hello there")) == 2


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


def test_scripted_curriculum_learns_all_initial_intents() -> None:
    agent = ConversationAgent(
        AgentConfig(seed=0, exploration_epsilon=0.15)
    )
    scenarios = default_scenarios()

    for _ in range(30):
        for scenario in scenarios:
            agent.train_response(scenario.messages[0], scenario.expected)

    results = [agent.respond(s.messages[0]).action for s in scenarios]

    assert results == [scenario.expected for scenario in scenarios]
