import numpy as np

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.cognition.working_memory import WorkingMemory


def test_working_memory_persists_and_decays_across_turns() -> None:
    memory = WorkingMemory(size=3, decay=0.5)
    memory.update(np.array([1.0, 0.0, 0.0]))
    memory.update(np.zeros(3))

    assert 0.0 < memory.state[0] < 1.0
    assert memory.turns == 2


def test_working_memory_reset_clears_context() -> None:
    memory = WorkingMemory(size=3)
    memory.update(np.ones(3))
    memory.reset()

    np.testing.assert_array_equal(memory.state, np.zeros(3))
    assert memory.turns == 0


def test_agent_retains_context_until_conversation_reset() -> None:
    agent = ConversationAgent(AgentConfig(input_features=16, seed=4))
    agent.respond("I am comparing two programming languages.")
    before_reset = agent.working_memory.state.copy()
    agent.respond("Which one should I choose?")
    after_second_turn = agent.working_memory.state.copy()

    assert np.any(before_reset != 0.0)
    assert np.any(after_second_turn != 0.0)
    assert agent.working_memory.turns == 2

    agent.reset_conversation()
    np.testing.assert_array_equal(agent.working_memory.state, np.zeros(16))


def test_working_memory_persists_without_voltage_readout_evidence() -> None:
    config = AgentConfig(input_features=16, seed=9)
    with_context = ConversationAgent(config)
    without_context = ConversationAgent(config)

    with_context.respond("I am comparing two programming languages.")
    contextual = with_context.respond("Which should I choose?")
    without = without_context.respond("Which should I choose?")

    assert np.any(with_context.working_memory.state != without_context.working_memory.state)
    assert contextual.timed_out and without.timed_out
