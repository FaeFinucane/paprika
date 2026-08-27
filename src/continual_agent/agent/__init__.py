"""Agent-level orchestration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_agent import ConversationAgent
    from .network_config import NetworkConfig
    from .spiking_runtime import SpikingRuntime


def __getattr__(name: str):
    if name == "ConversationAgent":
        from .conversation_agent import ConversationAgent

        return ConversationAgent
    if name == "SpikingRuntime":
        from .spiking_runtime import SpikingRuntime

        return SpikingRuntime
    if name == "NetworkConfig":
        from .network_config import NetworkConfig

        return NetworkConfig
    raise AttributeError(name)


__all__ = ["ConversationAgent", "NetworkConfig", "SpikingRuntime"]
