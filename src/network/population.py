"""Population specifications and their compiled neuron traits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Generic, Literal, Sequence, TypeVar

import numpy as np

OutputKind = Literal["excitatory", "inhibitory", "modulatory"]
DopamineResponse = Literal["aligned", "opposed", "neutral"]


@dataclass(frozen=True, slots=True)
class NeuronPopulationSpec:
    """A homogeneous ordinary-neuron population.

    Functional assemblies with mixed traits are represented by several named
    populations. This keeps transmitter sign and plasticity ownership visible
    in circuit construction rather than hidden in an offset convention.
    """

    name: str
    neuron_count: int
    output: OutputKind = "excitatory"
    dopamine_response: DopamineResponse = "neutral"
    kind: Literal["neurons"] = "neurons"

    def __post_init__(self) -> None:
        if self.neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if self.output not in ("excitatory", "inhibitory", "modulatory"):
            raise ValueError("invalid output kind")
        if self.dopamine_response not in ("aligned", "opposed", "neutral"):
            raise ValueError("invalid dopamine response")

    @property
    def count(self) -> int:
        return self.neuron_count


@dataclass(frozen=True, slots=True)
class FeaturePopulationSpec:
    name: str
    features: tuple[str, ...]
    feature_width: int
    kind: Literal["features"] = "features"

    def __post_init__(self) -> None:
        if (
            not self.features
            or len(set(self.features)) != len(self.features)
            or any(not value for value in self.features)
            or self.feature_width <= 0
        ):
            raise ValueError("invalid feature population")

    @property
    def count(self) -> int:
        return len(self.features) * self.feature_width

    @property
    def output(self) -> OutputKind:
        return "excitatory"

    @property
    def dopamine_response(self) -> DopamineResponse:
        return "neutral"

    def feature_bounds(self, feature: str) -> slice:
        index = self.features.index(feature)
        return slice(index * self.feature_width, (index + 1) * self.feature_width)


PopulationSpec = NeuronPopulationSpec | FeaturePopulationSpec
T = TypeVar("T", bound=PopulationSpec)


@dataclass(frozen=True, slots=True)
class Population(Generic[T]):
    spec: T
    start: int
    layout_fingerprint: str

    @property
    def bounds(self) -> slice:
        return slice(self.start, self.start + self.spec.count)

    @property
    def count(self) -> int:
        return self.spec.count


@dataclass(frozen=True)
class PopulationLayout:
    populations: tuple[Population[Any], ...]
    total_count: int
    fingerprint: str
    output_kinds: np.ndarray
    dopamine_response_sign: np.ndarray

    @staticmethod
    def build(specs: Sequence[PopulationSpec]) -> "PopulationLayout":
        if not specs:
            raise ValueError("at least one population is required")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("population names must be unique")

        fingerprint = sha256(repr(tuple(specs)).encode()).hexdigest()
        populations: list[Population[Any]] = []
        offset = 0
        for spec in specs:
            populations.append(Population(spec, offset, fingerprint))
            offset += spec.count

        output_kinds = np.empty(offset, dtype="U11")
        response = np.zeros(offset, dtype=float)
        response_values = {"aligned": 1.0, "opposed": -1.0, "neutral": 0.0}
        for population in populations:
            bounds = population.bounds
            spec = population.spec
            output_kinds[bounds] = spec.output
            response[bounds] = response_values[spec.dopamine_response]

        return PopulationLayout(
            tuple(populations),
            offset,
            fingerprint,
            output_kinds,
            response,
        )

    def population(self, name: str) -> Population[Any]:
        for population in self.populations:
            if population.spec.name == name:
                return population
        raise KeyError(f"unknown population {name!r}")

    @property
    def output_sign(self) -> np.ndarray:
        """Conventional-current sign. Modulatory neurons have no ordinary sign."""
        sign = np.zeros(self.total_count, dtype=float)
        sign[self.output_kinds == "excitatory"] = 1.0
        sign[self.output_kinds == "inhibitory"] = -1.0
        return sign
