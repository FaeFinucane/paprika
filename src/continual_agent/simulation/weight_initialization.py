"""Validated, reproducible initialization of network connectivity and weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from continual_agent.simulation.synapses import SparseSynapses


@dataclass(frozen=True)
class WeightInitializationConfig:
    """Distribution and bootstrap policy used while constructing a network."""

    excitatory_weight: float = 0.18
    inhibitory_weight: float = -0.12
    inhibitory_fraction: float = 0.15
    input_hidden_contacts: int = 3
    input_hidden_weight: float = 1.0
    input_hidden_spread: float = 0.0
    input_action_mean: float = 0.9
    input_action_spread: float = 0.08
    input_affect_mean: float = 0.0
    input_affect_spread: float = 0.02
    affect_action_mean: float = 0.0
    affect_action_spread: float = 0.02
    recurrent_output_mean: float = 0.0
    recurrent_output_spread: float = 0.025
    direct_output_mean: float = 0.0
    direct_output_spread: float = 0.025
    hidden_output_mean: float = 0.0
    hidden_output_spread: float = 0.025

    def __post_init__(self) -> None:
        if self.input_hidden_contacts <= 0:
            raise ValueError("input_hidden_contacts must be positive")
        if not 0.0 <= self.inhibitory_fraction <= 1.0:
            raise ValueError("inhibitory_fraction must be in [0, 1]")
        for name in (
            "input_hidden_spread",
            "input_action_spread",
            "input_affect_spread",
            "affect_action_spread",
            "recurrent_output_spread",
            "direct_output_spread",
            "hidden_output_spread",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "excitatory_weight",
            "inhibitory_weight",
            "input_hidden_weight",
            "input_action_mean",
            "input_affect_mean",
            "affect_action_mean",
            "recurrent_output_mean",
            "direct_output_mean",
            "hidden_output_mean",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
            if abs(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be within [-1, 1]")
        for mean, spread in (
            (self.input_hidden_weight, self.input_hidden_spread),
            (self.input_action_mean, self.input_action_spread),
            (self.input_affect_mean, self.input_affect_spread),
            (self.affect_action_mean, self.affect_action_spread),
            (self.recurrent_output_mean, self.recurrent_output_spread),
            (self.direct_output_mean, self.direct_output_spread),
            (self.hidden_output_mean, self.hidden_output_spread),
        ):
            if abs(mean) + spread > 1.0:
                raise ValueError("weight distributions must fit within [-1, 1]")


class WeightInitializer:
    """Apply a weight policy using stable, independent named RNG streams."""

    def __init__(self, seed: int, config: WeightInitializationConfig | None = None) -> None:
        self.seed = seed
        self.config = config or WeightInitializationConfig()

    def rng(self, name: str) -> np.random.Generator:
        # Keep the production connectivity graph stable while separating all
        # subsequent initialization decisions into named streams.
        if name == "connectivity":
            return np.random.default_rng(self.seed)
        digest = hashlib.blake2b(f"{self.seed}:{name}".encode("ascii"), digest_size=16).digest()
        return np.random.default_rng(np.frombuffer(digest, dtype=np.uint32))

    def seed_for(self, name: str) -> int:
        return int(self.rng(name).integers(0, np.iinfo(np.int64).max))

    def random_synapses(self, neuron_count: int, connection_probability: float) -> SparseSynapses:
        c = self.config
        return SparseSynapses.random(
            neuron_count,
            connection_probability,
            self.rng("connectivity"),
            excitatory_weight=c.excitatory_weight,
            inhibitory_weight=c.inhibitory_weight,
            inhibitory_fraction=c.inhibitory_fraction,
        )

    def projection(
        self,
        synapses: SparseSynapses,
        name: str,
        sources: np.ndarray,
        targets: np.ndarray,
        mean: float,
        spread: float,
    ) -> np.ndarray:
        source = np.repeat(sources, targets.size)
        weights = self.rng(name).normal(mean, spread, source.size)
        start = synapses.weight.size
        synapses.add_edges(source, np.tile(targets, sources.size), weights)
        return np.arange(start, synapses.weight.size)

    def bootstrap_input_hidden(
        self, synapses: SparseSynapses, sources: np.ndarray, targets: np.ndarray, name: str
    ) -> np.ndarray:
        c = self.config
        edges = [
            self.projection(
                synapses,
                f"{name}_{index}",
                sources,
                targets,
                c.input_hidden_weight,
                c.input_hidden_spread,
            )
            for index in range(c.input_hidden_contacts)
        ]
        return np.concatenate(edges) if edges else np.array([], dtype=np.int64)
