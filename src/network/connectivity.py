"""Setup and specs for synapses between neuron populations"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from .population import Population, PopulationLayout
import numpy as np

# TODO: Split out code for initializing weights from population & connectivity specs. Don't need initializer if we load from file. Initializer could even have info on loading from file.

@dataclass(frozen=True)
class BernoulliTopologySpec:
    expected_fan_out: float

    def __post_init__(self):
        if not np.isfinite(self.expected_fan_out) or self.expected_fan_out < 0:
            raise ValueError("fan-out must be finite and non-negative")

    def build_edges(self, source: Population[Any], target: Population[Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.expected_fan_out > target.count:
            raise ValueError("expected fan-out exceeds target population")
        
        probability = self.expected_fan_out / target.count
        src = np.repeat(np.arange(source.bounds.start, source.bounds.stop), target.count)
        dst = np.tile(np.arange(target.bounds.start, target.bounds.stop), source.count)
        keep = rng.random(src.size) < probability

        if source.spec.name == target.spec.name:
            keep &= src != dst

        return src[keep], dst[keep]


@dataclass(frozen=True)
class BimodalWeightSpec:
    positive_mean: float
    negative_mean: float
    negative_fraction: float
    positive_spread: float = 0.0
    negative_spread: float = 0.0
    minimum: float = -1.0
    maximum: float = 1.0

    def __post_init__(self):
        if (
            self.minimum > self.maximum
            or not 0 <= self.negative_fraction <= 1
            or min(self.positive_spread, self.negative_spread) < 0
        ):
            raise ValueError("invalid weight specification")

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        negative = rng.random(count) < self.negative_fraction
        result = np.where(
            negative,
            rng.normal(self.negative_mean, self.negative_spread, count),
            rng.normal(self.positive_mean, self.positive_spread, count),
        )
        return np.clip(result, self.minimum, self.maximum).astype(float)


@dataclass
class SparseSynapses:
    # TODO: Look into optimisations of this representation. It'll currently be non-sparse I think.
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    minimum: float = -1.0
    maximum: float = 1.0
    active: np.ndarray | None = None

    def __post_init__(self):
        if not (self.source.shape == self.target.shape == self.weight.shape):
            raise ValueError("synapse arrays must have equal shape")
        if self.active is None:
            self.active = np.ones(self.weight.size, dtype=bool)
        if self.active.shape != self.weight.shape:
            raise ValueError("active mask shape mismatch")
        if (
            not (np.isfinite(self.minimum) and np.isfinite(self.maximum))
            or self.minimum > self.maximum
        ):
            raise ValueError("invalid synapse weight limits")
        if not np.all(np.isfinite(self.weight)):
            raise ValueError("synapse weights must be finite")
        self.weight[:] = np.clip(self.weight, self.minimum, self.maximum)


@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    topology: BernoulliTopologySpec
    weight: BimodalWeightSpec

    @property
    def name(self):
        return f"{self.source}_to_{self.target}"


@dataclass(frozen=True)
class Connectivity:
    layout_fingerprint: str
    synapses: SparseSynapses
    edges: Mapping[str, np.ndarray]
    seed: int

    @staticmethod
    def build(layout: PopulationLayout, specs: Sequence[ConnectionSpec], seed: int):
        if len({(s.source, s.target) for s in specs}) != len(specs):
            raise ValueError("duplicate connection")
        
        sources: Sequence[np.ndarray] = []
        targets: Sequence[np.ndarray] = []
        weights: Sequence[np.ndarray] = []
        edges: Mapping[str, np.ndarray] = {}

        for spec in specs:
            s, t = layout.population(spec.source), layout.population(spec.target)
            rng = np.random.default_rng(
                [int(seed), int.from_bytes(spec.name.encode(), "little") % (2**32)]
            )
            a, b = spec.topology.build_edges(s, t, rng)
            w = spec.weight.sample(len(a), rng)
            start = sum(map(len, sources))
            sources.append(a)
            targets.append(b)
            weights.append(w)
            edges[spec.name] = np.arange(start, start + len(a))
        
        return Connectivity(
            layout.fingerprint,
            SparseSynapses(
                np.concatenate(sources) if sources else np.array([], int),
                np.concatenate(targets) if targets else np.array([], int),
                np.concatenate(weights) if weights else np.array([], float),
                *(next(iter(specs)).weight.minimum, next(iter(specs)).weight.maximum)
                if specs
                else (-1.0, 1.0),
            ),
            edges,
            int(seed),
        )
