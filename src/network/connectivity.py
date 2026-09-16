"""Declarative projections and compiled sparse synaptic strengths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from .population import Population, PopulationLayout

LearningPolicy = Literal["fixed", "dopamine_stdp", "inhibitory_homeostatic", "homeostatic"]


class TopologySpec(Protocol):
    def build_edges(
        self, source: Population[Any], target: Population[Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class FanOutSpec:
    expected: float
    target_neurons: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.expected) or self.expected < 0 or self.target_neurons < 0:
            raise ValueError("invalid fan-out")

    def build_edges(
        self, source: Population[Any], target: Population[Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target_neurons > target.count:
            raise ValueError("target_neurons exceeds target population")
        target_ids = (
            rng.choice(
                np.arange(target.bounds.start, target.bounds.stop), self.target_neurons, False
            )
            if self.target_neurons
            else np.arange(target.bounds.start, target.bounds.stop)
        )
        if self.expected > len(target_ids):
            raise ValueError("expected fan-out exceeds target population")
        source_ids = np.arange(source.bounds.start, source.bounds.stop)
        src = np.repeat(source_ids, len(target_ids))
        dst = np.tile(target_ids, len(source_ids))
        keep = (
            rng.random(src.size) < self.expected / len(target_ids) if len(target_ids) else src == -1
        )
        if source.spec.name == target.spec.name:
            keep &= src != dst
        return src[keep], dst[keep]


@dataclass(frozen=True)
class FanInSpec:
    expected: float
    source_neurons: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.expected) or self.expected < 0 or self.source_neurons < 0:
            raise ValueError("invalid fan-in")

    def build_edges(
        self, source: Population[Any], target: Population[Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.source_neurons > source.count:
            raise ValueError("source_neurons exceeds source population")
        source_ids = (
            rng.choice(
                np.arange(source.bounds.start, source.bounds.stop), self.source_neurons, False
            )
            if self.source_neurons
            else np.arange(source.bounds.start, source.bounds.stop)
        )
        if self.expected > len(source_ids):
            raise ValueError("expected fan-in exceeds source population")
        src = np.tile(source_ids, target.count)
        dst = np.repeat(np.arange(target.bounds.start, target.bounds.stop), len(source_ids))
        keep = (
            rng.random(src.size) < self.expected / len(source_ids) if len(source_ids) else src == -1
        )
        if source.spec.name == target.spec.name:
            keep &= src != dst
        return src[keep], dst[keep]


@dataclass(frozen=True)
class StrengthSpec:
    mean: float
    spread: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.mean)
            or self.spread < 0
            or self.minimum < 0
            or self.maximum < self.minimum
        ):
            raise ValueError("invalid strength specification")

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        return np.clip(rng.normal(self.mean, self.spread, count), self.minimum, self.maximum)


@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    topology: TopologySpec
    strength: StrengthSpec
    learning: LearningPolicy = "fixed"
    scalable: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if self.learning not in (
            "fixed",
            "dopamine_stdp",
            "inhibitory_homeostatic",
            "homeostatic",
        ):
            raise ValueError("invalid learning policy")
        if self.learning == "fixed" and self.scalable:
            raise ValueError("fixed projections cannot be scalable")

    @property
    def projection_name(self) -> str:
        return self.name or f"{self.source}_to_{self.target}"


@dataclass
class SparseSynapses:
    source: np.ndarray
    target: np.ndarray
    strength: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    learning: np.ndarray
    scalable: np.ndarray
    projection: np.ndarray
    active: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        arrays = (
            self.target,
            self.strength,
            self.minimum,
            self.maximum,
            self.learning,
            self.scalable,
            self.projection,
        )
        if any(array.shape != self.source.shape for array in arrays):
            raise ValueError("synapse arrays must have equal shape")
        if (
            not np.all(np.isfinite(self.strength))
            or not np.all(np.isfinite(self.minimum))
            or not np.all(np.isfinite(self.maximum))
            or np.any(self.minimum < 0)
            or np.any(self.maximum < self.minimum)
        ):
            raise ValueError("invalid synapse values")
        self.strength[:] = np.clip(self.strength, self.minimum, self.maximum)
        self.active = np.ones(self.source.size, dtype=bool)

    def policy_mask(self, policy: LearningPolicy) -> np.ndarray:
        return self.active & (self.learning == policy)

    def projection_mask(self, name: str) -> np.ndarray:
        return self.active & (self.projection == name)

    def validate_strength_delta(
        self, delta: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != self.strength.shape or not np.all(np.isfinite(delta)):
            raise ValueError("invalid strength delta")
        eligible = self.active & np.asarray(mask, dtype=bool)
        if eligible.shape != self.strength.shape:
            raise ValueError("invalid strength mask")
        if np.any(eligible & (self.learning == "fixed")):
            raise ValueError("fixed synapses cannot be updated")
        return delta, eligible

    def apply_strength_delta(self, delta: np.ndarray, mask: np.ndarray) -> None:
        delta, eligible = self.validate_strength_delta(delta, mask)
        self.strength[eligible] = np.clip(
            self.strength[eligible] + delta[eligible],
            self.minimum[eligible],
            self.maximum[eligible],
        )


def build_synapses(
    layout: PopulationLayout,
    specs: Sequence[ConnectionSpec],
    rng: np.random.Generator,
) -> SparseSynapses:
    names = [spec.projection_name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("projection names must be unique")

    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    strength_parts: list[np.ndarray] = []
    minimum_parts: list[np.ndarray] = []
    maximum_parts: list[np.ndarray] = []
    learning_parts: list[np.ndarray] = []
    scalable_parts: list[np.ndarray] = []
    projection_parts: list[np.ndarray] = []

    for spec in specs:
        source = layout.population(spec.source)
        target = layout.population(spec.target)
        if source.spec.output == "modulatory":
            raise ValueError("modulatory populations cannot make ordinary synaptic projections")
        if spec.learning == "inhibitory_homeostatic" and source.spec.output != "inhibitory":
            raise ValueError("inhibitory homeostasis requires an inhibitory source")
        src, dst = spec.topology.build_edges(source, target, rng)
        count = src.size
        source_parts.append(src)
        target_parts.append(dst)
        strength_parts.append(spec.strength.sample(count, rng))
        minimum_parts.append(np.full(count, spec.strength.minimum))
        maximum_parts.append(np.full(count, spec.strength.maximum))
        learning_parts.append(np.full(count, spec.learning, dtype="U22"))
        scalable_parts.append(np.full(count, spec.scalable, dtype=bool))
        projection_parts.append(np.full(count, spec.projection_name, dtype="U64"))

    def join(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    return SparseSynapses(
        join(source_parts, int),
        join(target_parts, int),
        join(strength_parts, float),
        join(minimum_parts, float),
        join(maximum_parts, float),
        join(learning_parts, "U22"),
        join(scalable_parts, bool),
        join(projection_parts, "U64"),
    )
