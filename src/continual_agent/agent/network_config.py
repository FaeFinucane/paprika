"""Construction of the concrete spiking network and its runtime services."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from continual_agent.agent.population_homeostasis import HomeostasisConfig
from continual_agent.agent.session import BOUNDARY_CHANNEL_COUNT, SessionPolicy
from continual_agent.simulation.weight_initialization import WeightInitializationConfig


@dataclass
class NetworkConfig:
    """Build layout, projections, state, drives, and scheduled plugins."""

    input_features: int = 48
    hidden_neurons: int = 48
    output_tokens: tuple[str, ...] | None = None
    neurons_per_token: int = 3
    action_names: tuple[str, ...] = ()
    neurons_per_action: int = 4
    affect_names: tuple[str, ...] = ()
    neurons_per_affect: int = 4
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
    language_alphabet: tuple[str, ...] = ("m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!")

    def __post_init__(self) -> None:
        if self.output_tokens is None:
            self.output_tokens = ("<EOS>",) + self.language_alphabet
        integer_fields = (
            "input_features",
            "hidden_neurons",
            "neurons_per_token",
            "neurons_per_action",
            "neurons_per_affect",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.output_tokens, tuple):
            raise ValueError("output_tokens must be a tuple")
        for name in ("action_names", "affect_names"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        if not isinstance(self.session_policy, SessionPolicy):
            raise ValueError("session_policy must be a SessionPolicy")
        if self.weight_initialization is not None and not isinstance(
            self.weight_initialization, WeightInitializationConfig
        ):
            raise ValueError("weight_initialization must be a WeightInitializationConfig or None")
        if self.input_features <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("input_features must leave room for boundary channels")
        if not self.output_tokens:
            raise ValueError("output_tokens must not be empty")
        if any(not isinstance(token, str) or not token for token in self.output_tokens):
            raise ValueError("output_tokens must contain non-empty strings")
        if "<EOS>" not in self.output_tokens:
            raise ValueError("output_tokens must contain <EOS>")
        for name, values in (
            ("action_names", self.action_names),
            ("affect_names", self.affect_names),
        ):
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain unique non-empty strings")
        if len(set(self.output_tokens)) != len(self.output_tokens):
            raise ValueError("output_tokens must be unique")
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
