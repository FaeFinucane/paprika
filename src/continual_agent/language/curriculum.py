"""Small-to-large character curricula for the spiking decoder."""

from __future__ import annotations

from dataclasses import dataclass

from continual_agent.cognition.readout import Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_agent.agent.conversation_agent import ConversationAgent


@dataclass(frozen=True)
class SequenceStage:
    name: str
    examples: tuple[str, ...]


def reduced_language_curriculum() -> tuple[SequenceStage, ...]:
    return (
        SequenceStage("sounds", ("m", "a", "b")),
        SequenceStage("transitions", ("ma", "ba", "am", "ab")),
        SequenceStage("repeated_syllables", ("mama", "baba", "maba", "bama")),
        SequenceStage("early_words", ("dada", "nana", "mama", "baba")),
    )


def train_reduced_curriculum(
    agent: "ConversationAgent",
    repetitions: int = 30,
) -> None:
    for stage in reduced_language_curriculum():
        for _ in range(repetitions):
            for example in stage.examples:
                agent.train_response_events(
                    Action.ANSWER,
                    agent.language.target_tokens(example),
                )
