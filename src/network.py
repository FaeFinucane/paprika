"""Deterministic compiled leaky integrate-and-fire network."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np

Array = np.ndarray


def _check_name(value: str, label: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True)
class NeuronPopulationSpec:
    name: str
    neuron_count: int
    kind: str = "neurons"

    def __post_init__(self):
        _check_name(self.name)
        if (
            self.kind != "neurons"
            or not isinstance(self.neuron_count, int)
            or self.neuron_count <= 0
        ):
            raise ValueError("invalid neuron population")


@dataclass(frozen=True)
class FeaturePopulationSpec:
    name: str
    features: tuple[str, ...]
    feature_width: int
    kind: str = "features"

    def __post_init__(self):
        _check_name(self.name)
        if (
            self.kind != "features"
            or not self.features
            or len(set(self.features)) != len(self.features)
            or any(not isinstance(x, str) or not x for x in self.features)
            or not isinstance(self.feature_width, int)
            or self.feature_width <= 0
        ):
            raise ValueError("invalid feature population")


@dataclass(frozen=True, slots=True)
class NeuronPopulation:
    name: str
    bounds: slice
    layout_fingerprint: str

    @property
    def count(self):
        return self.bounds.stop - self.bounds.start


@dataclass(frozen=True, slots=True)
class FeaturePopulation:
    name: str
    bounds: slice
    features: Mapping[str, slice]
    layout_fingerprint: str

    def __hash__(self):
        return hash((self.layout_fingerprint, self.name, self.bounds.start, self.bounds.stop))

    @property
    def count(self):
        return self.bounds.stop - self.bounds.start

    def feature_bounds(self, feature: str) -> slice:
        try:
            return self.features[feature]
        except KeyError:
            raise KeyError(f"unknown feature {feature!r}") from None


Population = NeuronPopulation | FeaturePopulation


@dataclass(frozen=True)
class PopulationLayout:
    populations: tuple[Population, ...]
    total_count: int
    fingerprint: str

    @staticmethod
    def build(specs: Sequence[NeuronPopulationSpec | FeaturePopulationSpec]):
        if not specs:
            raise ValueError("at least one population is required")
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("population names must be unique")
        raw = repr(tuple(specs)).encode()
        fp = sha256(raw).hexdigest()
        offset = 0
        pops = []
        for spec in specs:
            count = (
                spec.neuron_count
                if isinstance(spec, NeuronPopulationSpec)
                else len(spec.features) * spec.feature_width
            )
            bounds = slice(offset, offset + count)
            offset += count
            if isinstance(spec, FeaturePopulationSpec):
                features = {
                    f: slice(
                        bounds.start + i * spec.feature_width,
                        bounds.start + (i + 1) * spec.feature_width,
                    )
                    for i, f in enumerate(spec.features)
                }
                pops.append(FeaturePopulation(spec.name, bounds, features, fp))
            else:
                pops.append(NeuronPopulation(spec.name, bounds, fp))
        return PopulationLayout(tuple(pops), offset, fp)

    def population(self, name: str) -> Population:
        for p in self.populations:
            if p.name == name:
                return p
        raise KeyError(f"unknown population {name!r}")


@dataclass(frozen=True)
class BernoulliTopologySpec:
    expected_fan_out: float

    def __post_init__(self):
        if not np.isfinite(self.expected_fan_out) or self.expected_fan_out < 0:
            raise ValueError("fan-out must be finite and non-negative")

    def build_edges(self, source: Population, target: Population, rng: np.random.Generator):
        if self.expected_fan_out > target.count:
            raise ValueError("expected fan-out exceeds target population")
        probability = self.expected_fan_out / target.count
        src = np.repeat(np.arange(source.bounds.start, source.bounds.stop), target.count)
        dst = np.tile(np.arange(target.bounds.start, target.bounds.stop), source.count)
        keep = rng.random(src.size) < probability
        if source.name == target.name:
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

    def sample(self, count, rng):
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
    source: Array
    target: Array
    weight: Array
    minimum: float = -1.0
    maximum: float = 1.0
    active: Array | None = None

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
    edges: Mapping[str, Array]
    seed: int

    @staticmethod
    def build(layout, specs, seed):
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if len({(s.source, s.target) for s in specs}) != len(specs):
            raise ValueError("duplicate connection")
        sources = []
        targets = []
        weights = []
        edges = {}
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


@dataclass(frozen=True, slots=True)
class Spikes:
    values: Array
    tick: int

    def population(self, population):
        if population.layout_fingerprint != self._fingerprint:
            raise ValueError("population belongs to another layout")
        return self.values[population.bounds]

    _fingerprint: str = ""

    def __post_init__(self):
        if self.values.ndim != 1 or self.values.dtype != np.bool_:
            raise ValueError("spike values must be a one-dimensional boolean array")
        if self.tick < 0:
            raise ValueError("spike tick must be non-negative")


@dataclass
class LIFNeurons:
    voltage: Array
    refractory: Array
    threshold: float = 1.0
    decay: float = 0.9
    reset: float = 0.0
    refractory_ticks: int = 0

    @classmethod
    def build(cls, count, threshold=1.0, decay=0.9, reset=0.0, refractory_ticks=0):
        if threshold <= 0 or not 0 <= decay <= 1 or refractory_ticks < 0:
            raise ValueError("invalid LIF parameters")
        return cls(
            np.zeros(count), np.zeros(count, dtype=int), threshold, decay, reset, refractory_ticks
        )

    def step(self, current):
        self.voltage[self.refractory > 0] = self.reset
        self.refractory[self.refractory > 0] -= 1
        live = self.refractory == 0
        self.voltage[live] = self.decay * self.voltage[live] + current[live]
        spikes = live & (self.voltage >= self.threshold)
        self.voltage[spikes] = self.reset
        self.refractory[spikes] = self.refractory_ticks
        return spikes


@dataclass
class SNN:
    layout: PopulationLayout
    neurons: LIFNeurons
    synapses: SparseSynapses
    pending_current: Array
    tick: int = 0

    @staticmethod
    def build(layout, connectivity):
        if layout.fingerprint != connectivity.layout_fingerprint:
            raise ValueError("layout/connectivity mismatch")
        return SNN(
            layout,
            LIFNeurons.build(layout.total_count),
            connectivity.synapses,
            np.zeros(layout.total_count),
        )

    def step(self, external_current=None):
        current = self.pending_current.copy()
        self.pending_current.fill(0)
        for pop, value in (external_current or {}).items():
            value = np.asarray(value, dtype=float)
            if pop.layout_fingerprint != self.layout.fingerprint or value.shape != (pop.count,):
                raise ValueError("invalid external current")
            current[pop.bounds] += value
        emitted = self.neurons.step(current)
        if emitted.any():
            np.add.at(
                self.pending_current,
                self.synapses.target[self.synapses.active],
                self.synapses.weight[self.synapses.active]
                * emitted[self.synapses.source[self.synapses.active]],
            )
        self.tick += 1
        return Spikes(emitted.copy(), self.tick, self.layout.fingerprint)

    def apply_weight_delta(self, delta):
        delta = np.asarray(delta, dtype=float)
        if delta.shape != self.synapses.weight.shape or not np.all(np.isfinite(delta)):
            raise ValueError("invalid weight delta")
        active = self.synapses.active
        self.synapses.weight[active] = np.clip(
            self.synapses.weight[active] + delta[active],
            self.synapses.minimum,
            self.synapses.maximum,
        )
