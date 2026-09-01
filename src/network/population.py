"""Population specifications and representations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence

@dataclass(frozen=True)
class NeuronPopulationSpec:
    name: str
    neuron_count: int
    kind: Literal["neurons"] = "neurons"

    def __post_init__(self):
        if (
            self.neuron_count <= 0
        ):
            raise ValueError("invalid neuron population")


@dataclass(frozen=True)
class FeaturePopulationSpec:
    name: str
    features: tuple[str, ...]
    feature_width: int
    kind: Literal["features"] = "features"

    def __post_init__(self):
        if (
            not self.features
            or len(set(self.features)) != len(self.features)
            or any(not x for x in self.features)
            or self.feature_width <= 0
        ):
            raise ValueError("invalid feature population")


@dataclass(frozen=True, slots=True)
class NeuronPopulation:
    name: str
    bounds: slice[int]
    layout_fingerprint: str

    @property
    def count(self):
        return self.bounds.stop - self.bounds.start


@dataclass(frozen=True, slots=True)
class FeaturePopulation:
    name: str
    bounds: slice[int]
    # TODO: This should be easy enough to compute by itself.
    features: Mapping[str, slice[int]]
    layout_fingerprint: str

    def __hash__(self):
        return hash((self.layout_fingerprint, self.name, self.bounds.start, self.bounds.stop))

    @property
    def count(self):
        return self.bounds.stop - self.bounds.start

    def feature_bounds(self, feature: str) -> slice[int]:
        return self.features[feature]


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
        pops: Sequence[Population] = []
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