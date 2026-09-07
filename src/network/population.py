"""Population specifications and representations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Sequence

import numpy as np

# TODO: Consider moving population types up a level. Network layer only cares about number of neurons.
# TODO: Additionally look at ways we could abstract things so we don't need Spec + Population for everything.
# TODO: Maybe Population[PopulationSpec], with PopulationSpec being an ABC

@dataclass(frozen=True)
class NeuronPopulationSpec:
    name: str
    neuron_count: int
    # Proportion (0-1) of this population's neurons that are inhibitory.
    # Inhibitory neurons connect only to inhibitory synapses, see Dale's Principle.
    inhibitory: float = 0.0
    kind: Literal["neurons"] = "neurons"

    def __post_init__(self):
        if self.neuron_count <= 0:
            raise ValueError("invalid neuron population")
        if not 0 <= self.inhibitory <= 1:
            raise ValueError("inhibitory must be a proportion between 0 and 1")

    @property
    def count(self):
        return self.neuron_count

    @property
    def inhibitory_count(self) -> int:
        return round(self.neuron_count * self.inhibitory)


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

    @property
    def count(self):
        return len(self.features) * self.feature_width

    @property
    def inhibitory_count(self) -> int:
        return 0

    def feature_bounds(self, feature: str) -> slice[int]:
        index = self.features.index(feature)
        return slice(
            index * self.feature_width,
            (index + 1) * self.feature_width,
        )

@dataclass(frozen=True, slots=True)
class NumericPopulationSpec:
    name: str
    positive: int
    negative: int
    kind: Literal["numeric"] = "numeric"

    @property
    def count(self):
        return self.positive + self.negative

    @property
    def inhibitory_count(self) -> int:
        return 0

# Population = NeuronPopulation | FeaturePopulation | NumericPopulation
PopulationSpec = NeuronPopulationSpec | FeaturePopulationSpec | NumericPopulationSpec

@dataclass(frozen=True, slots=True)
class Population[T: PopulationSpec]:
    spec: T
    start: int
    layout_fingerprint: str

    @property
    def bounds(self) -> slice[int]:
        return slice(self.start, self.start + self.spec.count)

    @property
    def count(self) -> int:
        return self.spec.count

# TODO: Consider build returning a dict of populations directly, to look up by name, as well as the layout itself.
@dataclass(frozen=True)
class PopulationLayout:
    populations: tuple[Population[Any], ...]
    total_count: int
    fingerprint: str

    @staticmethod
    def build(specs: Sequence[NeuronPopulationSpec | FeaturePopulationSpec | NumericPopulationSpec]):
        if not specs:
            raise ValueError("at least one population is required")
        
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("population names must be unique")
        
        raw = repr(tuple(specs)).encode()
        fp = sha256(raw).hexdigest()

        offset = 0
        pops: Sequence[Population[Any]] = []
        for spec in specs:
            bounds = slice(offset, offset + spec.count)
            offset = bounds.stop
            pops.append(Population(spec, bounds.start, fp))
        return PopulationLayout(tuple(pops), offset, fp)

    def population(self, name: str) -> Population[Any]:
        for p in self.populations:
            if p.spec.name == name:
                return p
        raise KeyError(f"unknown population {name!r}")

    def neuron_types(self) -> np.ndarray[tuple[int],np.dtype[np.bool]]:
        """Boolean mask of whether neurons are excitatory (false) or inhibitory (true)"""
        mask = np.zeros(self.total_count, dtype=bool)
        for p in self.populations:
            if p.spec.inhibitory_count:
                mask[p.bounds.stop - p.spec.inhibitory_count:p.bounds.stop] = True
        return mask