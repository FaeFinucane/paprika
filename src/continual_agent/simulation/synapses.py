"""Sparse synaptic connectivity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SYNAPTIC_WEIGHT_LIMIT = 1.0


@dataclass(frozen=True)
class SynapticActivityState:
    """Initial synaptic activity kept separately from synaptic weights.

    ``update`` uses a weighted, decayed average: the prior observation receives
    ``decay * sample_weight`` and the new observation receives ``weight``.
    This preserves a bounded activity value while retaining recency information.
    """

    activity: np.ndarray
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        activity = np.asarray(self.activity, dtype=float)
        if activity.ndim != 1 or not np.isfinite(activity).all():
            raise ValueError("synaptic activity must be a finite 1D array")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample_weight must be positive and finite")
        object.__setattr__(self, "activity", activity.copy())

    def copy(self) -> "SynapticActivityState":
        return SynapticActivityState(self.activity, self.sample_weight)

    def update(
        self, observation: np.ndarray, weight: float = 1.0, decay: float = 0.9
    ) -> "SynapticActivityState":
        observation = np.asarray(observation, dtype=float)
        if observation.shape != self.activity.shape:
            raise ValueError("activity observation has the wrong shape")
        if not np.isfinite(observation).all() or not 0 <= decay <= 1:
            raise ValueError("observation must be finite and decay must be in [0, 1]")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be positive and finite")
        prior = decay * self.sample_weight
        total = prior + weight
        return SynapticActivityState(
            (prior * self.activity + weight * observation) / total,
            total,
        )

    @classmethod
    def aggregate(
        cls,
        observations: list[np.ndarray] | tuple[np.ndarray, ...],
        weights: np.ndarray | None = None,
        decay: float = 0.9,
    ) -> "SynapticActivityState":
        """Aggregate observations with recency decay, oldest to newest."""

        if not observations:
            raise ValueError("at least one activity observation is required")
        if not 0 <= decay <= 1:
            raise ValueError("decay must be in [0, 1]")
        values = [np.asarray(item, dtype=float) for item in observations]
        if any(item.ndim != 1 or not np.isfinite(item).all() for item in values):
            raise ValueError("activity observations must be finite 1D arrays")
        if any(item.shape != values[0].shape for item in values[1:]):
            raise ValueError("activity observations must have equal shapes")
        if weights is None:
            factors = np.ones(len(values), dtype=float)
        else:
            factors = np.asarray(weights, dtype=float)
            if (
                factors.shape != (len(values),)
                or not np.isfinite(factors).all()
                or np.any(factors <= 0)
            ):
                raise ValueError("weights must be positive and match observations")
        factors *= decay ** np.arange(len(values) - 1, -1, -1)
        return cls(np.average(np.stack(values), axis=0, weights=factors), float(factors.sum()))


@dataclass
class SparseSynapses:
    """Directed sparse connections stored as source/target/weight arrays."""

    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    neuron_count: int
    _connectivity_frozen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.source = np.array(self.source, dtype=np.int64, copy=True)
        self.target = np.array(self.target, dtype=np.int64, copy=True)
        self.weight = np.array(self.weight, dtype=float, copy=True)
        if not (self.source.shape == self.target.shape == self.weight.shape):
            raise ValueError("source, target, and weight must have equal shapes")
        if self.neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if np.any(self.source < 0) or np.any(self.source >= self.neuron_count):
            raise ValueError("source contains an invalid neuron index")
        if np.any(self.target < 0) or np.any(self.target >= self.neuron_count):
            raise ValueError("target contains an invalid neuron index")
        if (
            np.unique(np.stack((self.source, self.target), axis=1), axis=0).shape[0]
            != self.source.size
        ):
            raise ValueError("duplicate directed synaptic edges are not allowed")
        np.clip(
            self.weight,
            -SYNAPTIC_WEIGHT_LIMIT,
            SYNAPTIC_WEIGHT_LIMIT,
            out=self.weight,
        )

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
        self.transmit_into(spikes, current)
        return current

    def transmit_into(self, spikes: np.ndarray, current: np.ndarray) -> None:
        """Write next-tick current into a caller-owned buffer."""
        spikes = np.asarray(spikes, dtype=bool)
        if spikes.shape != (self.neuron_count,) or current.shape != (self.neuron_count,):
            raise ValueError(f"spikes and current must have shape ({self.neuron_count},)")
        current.fill(0.0)
        active_edges = spikes[self.source]
        np.add.at(current, self.target[active_edges], self.weight[active_edges])

    def add_edges(self, source: np.ndarray, target: np.ndarray, weight: np.ndarray) -> None:
        """Append connections atomically, useful for explicit projections."""

        if self._connectivity_frozen:
            raise RuntimeError("synaptic connectivity is immutable after network construction")

        extra = SparseSynapses(source, target, weight, self.neuron_count)
        existing = np.stack((self.source, self.target), axis=1)
        added = np.stack((extra.source, extra.target), axis=1)
        if (
            np.unique(np.concatenate((existing, added)), axis=0).shape[0]
            != existing.shape[0] + added.shape[0]
        ):
            raise ValueError("duplicate directed synaptic edges are not allowed")
        source = np.concatenate((self.source, extra.source))
        target = np.concatenate((self.target, extra.target))
        weight = np.concatenate((self.weight, extra.weight))
        self.source, self.target, self.weight = source, target, weight

    def freeze_connectivity(self) -> None:
        """Prevent edge-count changes after edge-indexed runtime state exists."""
        self._connectivity_frozen = True
