"""Agent-level orchestration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_agent import ConversationAgent
    from .spiking_runtime import SpikingRuntime


def __getattr__(name: str):
    if name == "ConversationAgent":
        from .conversation_agent import ConversationAgent

        return ConversationAgent
    if name == "SpikingRuntime":
        from .spiking_runtime import SpikingRuntime

        return SpikingRuntime
    raise AttributeError(name)


__all__ = ["ConversationAgent", "SpikingRuntime"]
