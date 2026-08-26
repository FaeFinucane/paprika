"""Agent-level orchestration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_agent import ConversationAgent


def __getattr__(name: str):
    if name == "ConversationAgent":
        from .conversation_agent import ConversationAgent

        return ConversationAgent
    raise AttributeError(name)

__all__ = ["ConversationAgent"]
