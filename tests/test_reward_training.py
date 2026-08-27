import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.agent.session import InputSignal
from continual_agent.simulation.population_layout import Population


def test_reward_modulated_training_commits_one_diffuse_reward() -> None:
    agent = ConversationAgent()
    frame = np.zeros(agent.runtime.layout.slice(Population.INPUT).stop)
    frame[1] = 1.0
    agent.runtime.train_reward_modulated_events(
        (InputSignal.INPUT_BEGIN, (frame, "a"), InputSignal.INPUT_END),
        targets=("a", "<EOS>"),
    )

    assert agent.runtime.response_session.state.value == "idle"


def test_reward_modulated_training_rejects_unbounded_input() -> None:
    agent = ConversationAgent()

    try:
        agent.runtime.train_reward_modulated_events((InputSignal.INPUT_BEGIN,))
    except ValueError as error:
        assert "boundaries" in str(error)
    else:
        raise AssertionError("missing input boundary was accepted")

    assert agent.runtime.response_session.state.value == "idle"
