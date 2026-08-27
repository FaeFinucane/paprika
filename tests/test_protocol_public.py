import math

import pytest

from continual_agent.cognition.readout import Action
from continual_agent.environment.protocol import AgentAction, Feedback, UserMessage


def test_environment_messages_and_actions_are_validated() -> None:
    assert UserMessage("hello").text == "hello"
    assert Feedback(1.0, Action.ANSWER).expected is Action.ANSWER
    assert AgentAction(Action.ANSWER, 0.5, 3).thinking_ticks == 3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UserMessage(""),
        lambda: Feedback(math.nan, Action.ANSWER),
        lambda: AgentAction(Action.ANSWER, 1.1, 0),
        lambda: AgentAction(Action.ANSWER, 0.5, -1),
    ],
)
def test_environment_protocol_rejects_invalid_boundaries(factory) -> None:
    with pytest.raises(ValueError):
        factory()
