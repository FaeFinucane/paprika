"""Setup and specs for synapses between neuron populations"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .population import Population, PopulationLayout
import numpy as np

# TODO: Split out code for initializing weights from population & connectivity specs. Don't need initializer if we load from file. Initializer could even have info on loading from file.


class TopologySpec(Protocol):
    def build_edges(self, source: Population[Any], target: Population[Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]: ...

@dataclass(frozen=True)
class FanOutSpec(TopologySpec):
    """Connects each source neuron to `expected` target neurons"""

    expected: float
    target_neurons: int = 0

    def __post_init__(self):
        if not np.isfinite(self.expected) or self.expected < 0:
            raise ValueError("fan-out must be finite and non-negative")
        if self.target_neurons < 0:
            raise ValueError("target_neurons must be non-negative")

    def build_edges(self, source: Population[Any], target: Population[Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.target_neurons > 0:
            if self.target_neurons > target.count:
                raise ValueError("target_neurons exceeds target population")
            target_ids = rng.choice(np.arange(target.bounds.start, target.bounds.stop), size=self.target_neurons, replace=False)
            pool_size = self.target_neurons
        else:
            target_ids = np.arange(target.bounds.start, target.bounds.stop)
            pool_size = target.count

        if self.expected > pool_size:
            raise ValueError("expected fan-out exceeds target population")

        probability = self.expected / pool_size
        src = np.repeat(np.arange(source.bounds.start, source.bounds.stop), pool_size)
        dst = np.tile(target_ids, source.count)
        keep = rng.random(src.size) < probability

        if source.spec.name == target.spec.name:
            keep &= src != dst

        return src[keep], dst[keep]


@dataclass(frozen=True)
class FanInSpec(TopologySpec):
    """Connects `expected` source neurons to each target neuron"""

    expected: float
    source_neurons: int = 0

    def __post_init__(self):
        if not np.isfinite(self.expected) or self.expected < 0:
            raise ValueError("fan-in must be finite and non-negative")
        if self.source_neurons < 0:
            raise ValueError("source_neurons must be non-negative")

    def build_edges(self, source: Population[Any], target: Population[Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.source_neurons > 0:
            if self.source_neurons > source.count:
                raise ValueError("source_neurons exceeds source population")
            source_ids = rng.choice(np.arange(source.bounds.start, source.bounds.stop), size=self.source_neurons, replace=False)
            pool_size = self.source_neurons
        else:
            source_ids = np.arange(source.bounds.start, source.bounds.stop)
            pool_size = source.count

        if self.expected > pool_size:
            raise ValueError("expected fan-in exceeds source population")

        probability = self.expected / pool_size
        src = np.tile(source_ids, target.count)
        dst = np.repeat(np.arange(target.bounds.start, target.bounds.stop), pool_size)
        keep = rng.random(src.size) < probability

        if source.spec.name == target.spec.name:
            keep &= src != dst

        return src[keep], dst[keep]


@dataclass(frozen=True)
class WeightSpec:
    mean: float
    spread: float = 0.0

    def __post_init__(self):
        if self.spread < 0:
            raise ValueError("invalid weight specification")

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        return rng.normal(self.mean, self.spread, count).astype(float)


@dataclass
class SparseSynapses:
    # TODO: Look into optimisations of this representation. It'll currently be non-sparse I think.
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    minimum: float = -1.0
    maximum: float = 1.0
    active: np.ndarray = field(init = False)

    def __post_init__(self):
        if not (self.source.shape == self.target.shape == self.weight.shape):
            raise ValueError("synapse arrays must have equal shape")
        self.active = np.ones(self.weight.size, dtype=bool)
        if (
            not (np.isfinite(self.minimum) and np.isfinite(self.maximum))
            or self.minimum > self.maximum
        ):
            raise ValueError("invalid synapse weight limits")
        if not np.all(np.isfinite(self.weight)):
            raise ValueError("synapse weights must be finite")
        self.weight[:] = np.clip(self.weight, self.minimum, self.maximum)

    def renormalize(self, layout: PopulationLayout, population: Population[Any], total: float):
        """Renormalize all neuron input synapse weights to equal `total`"""
        neuron_types = layout.neuron_types()
        in_population = (self.target >= population.bounds.start) & (self.target < population.bounds.stop)
        excitatory = in_population & ~neuron_types[self.source]

        totals = np.bincount(self.target[excitatory], weights=self.weight[excitatory], minlength=neuron_types.shape[0])
        scale = np.ones_like(totals)
        nonzero = totals > 0
        scale[nonzero] = total / totals[nonzero]

        self.weight[excitatory] *= scale[self.target[excitatory]]

@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    topology: TopologySpec
    weight: WeightSpec

    @property
    def name(self):
        return f"{self.source}_to_{self.target}"


def build_synapses(
    layout: PopulationLayout,
    specs: Sequence[ConnectionSpec],
    rng: np.random.Generator,
    maximum: float = 1.0,
    minimum: float = -1.0,
) -> SparseSynapses:
    if len({(s.source, s.target) for s in specs}) != len(specs):
        raise ValueError("duplicate connection")

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    inhibitory_mask = layout.neuron_types()

    for spec in specs:
        s, t = layout.population(spec.source), layout.population(spec.target)
        a, b = spec.topology.build_edges(s, t, rng)
        w = spec.weight.sample(len(a), rng)
        w = np.where(inhibitory_mask[a], -w, w)

        sources.append(a)
        targets.append(b)
        weights.append(w)

    return SparseSynapses(
        np.concatenate(sources) if sources else np.array([], int),
        np.concatenate(targets) if targets else np.array([], int),
        np.concatenate(weights) if weights else np.array([], float),
        minimum,
        maximum,
    )
