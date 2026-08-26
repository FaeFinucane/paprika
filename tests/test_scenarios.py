from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.environment.scenarios import default_scenarios
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
