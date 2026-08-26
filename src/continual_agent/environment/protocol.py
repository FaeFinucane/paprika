"""Typed event/action boundaries used by the conversation laboratory."""

from __future__ import annotations

from dataclasses import dataclass

from continual_agent.cognition.readout import Action


@dataclass(frozen=True)
class UserMessage:
    text: str
    end_conversation: bool = False


@dataclass(frozen=True)
class Feedback:
    reward: float
    expected: Action


@dataclass(frozen=True)
class AgentAction:
    action: Action
    confidence: float
    thinking_ticks: int
