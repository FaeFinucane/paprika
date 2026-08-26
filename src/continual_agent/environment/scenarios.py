"""Templated conversation scenarios with machine-checkable outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from continual_agent.cognition.readout import Action
from continual_agent.cognition.affect import AffectiveEvent


@dataclass(frozen=True)
class ConversationScenario:
    name: str
    messages: tuple[str, ...]
    expected: Action
    affect_event: AffectiveEvent = AffectiveEvent()
    affect_targets: dict[str, tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.affect_targets is None:
            object.__setattr__(self, "affect_targets", {})

    def reward_for(self, action: Action) -> float:
        return 1.0 if action == self.expected else -1.0


def default_scenarios() -> tuple[ConversationScenario, ...]:
    return (
        ConversationScenario(
            "ambiguous_request",
            ("Can you help me choose?",),
            Action.CLARIFY,
            AffectiveEvent(uncertainty=0.9, novelty=0.2),
            {"uncertainty": (0.5, 1.0)},
        ),
        ConversationScenario(
            "unknown_information",
            ("What is the answer to a question with no known context?",),
            Action.UNCERTAIN,
            AffectiveEvent(uncertainty=1.0, novelty=0.1),
            {"uncertainty": (0.5, 1.0)},
        ),
        ConversationScenario(
            "simple_request",
            ("Please explain this clearly.",),
            Action.ANSWER,
            AffectiveEvent(reward_prediction_error=0.4),
            {"valence": (0.0, 1.0), "competence": (0.5, 1.0)},
        ),
        ConversationScenario(
            "user_correction",
            ("That is not what I meant.",),
            Action.REVISE,
            AffectiveEvent(correction=True, uncertainty=0.9),
            {"uncertainty": (0.5, 1.0)},
        ),
        ConversationScenario(
            "positive_acknowledgement",
            ("Thank you, that helps.",),
            Action.ACKNOWLEDGE,
            AffectiveEvent(social_feedback=1.0, reward_prediction_error=0.4),
            {"social_affiliation": (0.5, 1.0), "valence": (0.0, 1.0)},
        ),
    )
