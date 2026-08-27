"""Agent-level orchestration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network_config import NetworkConfig
    from .spiking_runtime import SpikingRuntime


def __getattr__(name: str):
    if name == "SpikingRuntime":
        from .spiking_runtime import SpikingRuntime

        return SpikingRuntime
    if name == "NetworkConfig":
        from .network_config import NetworkConfig

        return NetworkConfig
    raise AttributeError(name)


__all__ = ["NetworkConfig", "SpikingRuntime"]
