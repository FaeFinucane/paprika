import pytest

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.readout import Action
from continual_agent.environment.scenarios import ConversationScenario, default_scenarios
from continual_agent.evaluation.metrics import evaluate_affect_targets


def test_default_scenarios_have_machine_checkable_affect_targets() -> None:
    scenarios = default_scenarios()

    assert all(scenario.affect_targets for scenario in scenarios)
    assert all(scenario.affect_event is not None for scenario in scenarios)


def test_ambiguous_scenario_raises_uncertainty() -> None:
    agent = ConversationAgent()
    scenario = default_scenarios()[0]

    result = evaluate_affect_targets(agent, scenario)

    assert result["uncertainty"]


def test_affect_evaluation_does_not_mutate_agent() -> None:
    agent = ConversationAgent()
    before = agent.affect.as_dict()
    evaluate_affect_targets(agent, default_scenarios()[0])
    assert agent.affect.as_dict() == before


def test_scenario_reward_is_action_specific() -> None:
    scenario = default_scenarios()[0]
    assert scenario.reward_for(scenario.expected) == 1.0
    assert scenario.reward_for(Action.ANSWER) == -1.0


@pytest.mark.parametrize("targets", [{"valence": (2.0, 1.0)}, {"valence": (float("nan"), 1.0)}])
def test_scenario_rejects_invalid_affect_targets(targets: dict[str, tuple[float, float]]) -> None:
    with pytest.raises(ValueError):
        ConversationScenario("bad", ("hello",), Action.ANSWER, affect_targets=targets)
