"""Training and evaluation helpers for the first vertical slice."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from continual_agent.environment.scenarios import ConversationScenario

if TYPE_CHECKING:
    from continual_agent.agent.conversation_agent import ConversationAgent


def evaluate_affect_targets(
    agent: ConversationAgent,
    scenario: ConversationScenario,
) -> dict[str, bool]:
    """Check whether the post-event affective state meets scenario ranges."""

    if scenario.affect_event is None:
        raise ValueError("scenario must define an affect event")
    isolated = deepcopy(agent.affect)
    isolated.observe(scenario.affect_event)
    values = isolated.as_dict()
    return {
        name: name in values and lower <= values[name] <= upper
        for name, (lower, upper) in scenario.affect_targets.items()
    }
