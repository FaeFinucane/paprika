"""Run the first automated conversation-learning experiment."""

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.environment.scenarios import default_scenarios
from continual_agent.evaluation.metrics import run_curriculum


def main() -> None:
    agent = ConversationAgent()
    scenarios = default_scenarios()
    report = run_curriculum(agent, scenarios, repetitions=30)
    print(f"interactions: {len(report.rewards)}")
    print(f"mean reward: {report.mean_reward:.3f}")
    for scenario in scenarios:
        decision = agent.respond(scenario.messages[0])
        print(f"{scenario.name}: {decision.action.value} ({decision.confidence:.2f})")


if __name__ == "__main__":
    main()
