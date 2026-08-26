"""Sparse synaptic connectivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseSynapses:
    """Directed sparse connections stored as source/target/weight arrays."""

    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    neuron_count: int

    def __post_init__(self) -> None:
        self.source = np.asarray(self.source, dtype=np.int64)
        self.target = np.asarray(self.target, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=float)
        if not (self.source.shape == self.target.shape == self.weight.shape):
            raise ValueError("source, target, and weight must have equal shapes")
        if self.neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if np.any(self.source < 0) or np.any(self.source >= self.neuron_count):
            raise ValueError("source contains an invalid neuron index")
        if np.any(self.target < 0) or np.any(self.target >= self.neuron_count):
            raise ValueError("target contains an invalid neuron index")

    @classmethod
    def random(
        cls,
        neuron_count: int,
        connection_probability: float,
        rng: np.random.Generator,
        excitatory_weight: float = 0.25,
        inhibitory_weight: float = -0.25,
        inhibitory_fraction: float = 0.2,
    ) -> "SparseSynapses":
        """Create deterministic sparse connectivity from an explicit RNG."""

        if not 0 <= connection_probability <= 1:
            raise ValueError("connection_probability must be in [0, 1]")
        if not 0 <= inhibitory_fraction <= 1:
            raise ValueError("inhibitory_fraction must be in [0, 1]")

        mask = rng.random((neuron_count, neuron_count)) < connection_probability
        np.fill_diagonal(mask, False)
        source, target = np.nonzero(mask)
        inhibitory = rng.random(source.size) < inhibitory_fraction
        weights = np.where(inhibitory, inhibitory_weight, excitatory_weight)
        return cls(source, target, weights, neuron_count)

    def transmit(self, spikes: np.ndarray) -> np.ndarray:
        """Return currents delivered on the next tick by ``spikes``."""

        spikes = np.asarray(spikes, dtype=bool)
        if spikes.shape != (self.neuron_count,):
            raise ValueError(f"spikes must have shape ({self.neuron_count},)")

        current = np.zeros(self.neuron_count, dtype=float)
        active_edges = spikes[self.source]
        np.add.at(current, self.target[active_edges], self.weight[active_edges])
        return current

    def add_edges(
        self, source: np.ndarray, target: np.ndarray, weight: np.ndarray
    ) -> None:
        """Append connections, useful for explicit sensory projections."""

        extra = SparseSynapses(source, target, weight, self.neuron_count)
        self.source = np.concatenate((self.source, extra.source))
        self.target = np.concatenate((self.target, extra.target))
        self.weight = np.concatenate((self.weight, extra.weight))
