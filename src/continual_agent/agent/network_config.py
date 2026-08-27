"""Construction of the concrete spiking network and its runtime services."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from continual_agent.agent.population_homeostasis import HomeostasisConfig
from continual_agent.agent.session import BOUNDARY_CHANNEL_COUNT, SessionPolicy
from continual_agent.simulation.population_layout import PopulationLayout
from continual_agent.simulation.weight_initialization import WeightInitializationConfig


@dataclass
class NetworkConfig:
    """Finalized population layout plus simulation and training settings."""

    layout: PopulationLayout = field(default_factory=PopulationLayout.from_dimensions)
    connection_probability: float = 0.08
    seed: int = 0
    weight_initialization: WeightInitializationConfig | None = None
    learning_rate: float = 0.08
    background_rate: float = 0.0
    background_current: float = 0.05
    session_policy: SessionPolicy = field(default_factory=SessionPolicy)
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    max_thinking_ticks: int = 18
    max_response_tokens: int = 48
    max_response_ticks: int = 144
    presentation_speed: int = 1
    persistent_working_memory: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout, PopulationLayout):
            raise ValueError("layout must be a PopulationLayout")
        if self.layout.input_count <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("layout input population must leave room for boundary channels")
        output_tokens = tuple(self.layout.char_subgroups)
        if not output_tokens or "<EOS>" not in output_tokens:
            raise ValueError("layout character subgroups must contain <EOS>")
        if not isinstance(self.session_policy, SessionPolicy):
            raise ValueError("session_policy must be a SessionPolicy")
        if self.weight_initialization is not None and not isinstance(
            self.weight_initialization, WeightInitializationConfig
        ):
            raise ValueError("weight_initialization must be a WeightInitializationConfig or None")
        if not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        numeric_fields = (
            "connection_probability",
            "learning_rate",
            "background_rate",
            "background_current",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.connection_probability <= 1:
            raise ValueError("connection_probability must be in [0, 1]")
        if not 0 <= self.background_rate <= 1 or self.background_current < 0:
            raise ValueError("background rate must be in [0, 1] and current must be non-negative")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not isinstance(self.homeostasis, HomeostasisConfig):
            raise ValueError("homeostasis must be a HomeostasisConfig")
