"""Training and evaluation helpers for the first vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.environment.scenarios import ConversationScenario


@dataclass
class TrainingReport:
    rewards: list[float] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0


def run_curriculum(
    agent: ConversationAgent,
    scenarios: tuple[ConversationScenario, ...],
    repetitions: int,
) -> TrainingReport:
    report = TrainingReport()
    for _ in range(repetitions):
        for scenario in scenarios:
            result = agent.train_response(scenario.messages[0], scenario.expected)
            reward = 1.0 if result.action == scenario.expected else -1.0
            report.rewards.append(reward)
            report.actions.append(result.action.value)
    return report


def evaluate_affect_targets(
    agent: ConversationAgent,
    scenario: ConversationScenario,
) -> dict[str, bool]:
    """Check whether the post-event affective state meets scenario ranges."""

    agent.affect.observe(scenario.affect_event)
    values = agent.affect.as_dict()
    return {
        name: lower <= values[name] <= upper
        for name, (lower, upper) in scenario.affect_targets.items()
    }
