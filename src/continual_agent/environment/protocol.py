"""Typed event/action boundaries used by the conversation laboratory."""

from __future__ import annotations

import math
from dataclasses import dataclass

from continual_agent.cognition.readout import Action


@dataclass(frozen=True)
class UserMessage:
    text: str
    end_conversation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True)
class Feedback:
    reward: float
    expected: Action

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.expected, Action):
            raise ValueError("expected must be an Action")


@dataclass(frozen=True)
class AgentAction:
    action: Action
    confidence: float
    thinking_ticks: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ValueError("action must be an Action")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if isinstance(self.thinking_ticks, bool) or self.thinking_ticks < 0:
            raise ValueError("thinking_ticks must be non-negative")
