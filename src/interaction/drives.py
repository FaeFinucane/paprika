"""Explicit ongoing-current sources for otherwise input-driven networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..builder import PopulationPluginSpec
from ..network.adjustments import NetworkAdjustment
from ..network.population import Population
from ..network.snn import SNN, Spikes
from ..session import Session
from .plugins import Drives, DriveSource, Hook, StatefulAdaptation


@dataclass
class TonicDrive(DriveSource):
    """A named tonic current source with optional fixed cell diversity."""

    population: Population[Any]
    spec: TonicDriveSpec
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    _currents: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.spec.current)
            or not np.isfinite(self.spec.heterogeneity)
            or self.spec.heterogeneity < 0
        ):
            raise ValueError("tonic current must be finite and heterogeneity non-negative")
        variation = self.rng.normal(0.0, self.spec.heterogeneity, self.population.count)
        variation -= float(np.mean(variation))
        self._currents = self.spec.current + variation

    @property
    def current(self) -> float:
        """Mean configured tonic current for diagnostic reporting."""
        return self.spec.current

    def produce(self) -> Drives:
        return Drives({self.population: self._currents})


@dataclass
class HomeostaticDrive(DriveSource, StatefulAdaptation):
    """Slow, bounded rate regulation through an explicit current source.

    The controller owns its current rather than changing hidden neuron
    parameters. Its output is consequently visible in circuit construction and
    can be removed from a session to test the recurrent circuit on its own.
    """

    population: Population[Any]
    spec: HomeostaticDriveSpec
    _rate: float = field(default=0.0, init=False, repr=False)
    _current: float = field(init=False, repr=False)
    _proposed_current: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not 0 <= self.spec.target_rate <= 1
            or self.spec.learning_rate < 0
            or not 0 <= self.spec.rate_decay < 1
            or not np.isfinite(self.spec.minimum_current)
            or not np.isfinite(self.spec.maximum_current)
            or self.spec.minimum_current > self.spec.maximum_current
            or not self.spec.minimum_current
            <= self.spec.initial_current
            <= self.spec.maximum_current
        ):
            raise ValueError("invalid homeostatic-drive configuration")
        self._current = self.spec.initial_current

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def current(self) -> float:
        """Current emitted on the next tick."""
        return self._current

    def produce(self) -> Drives:
        return Drives({self.population: np.full(self.population.count, self._current)})

    def observe(self, spikes: Spikes) -> None:
        observed = float(np.mean(spikes.population(self.population)))
        self._rate = self.spec.rate_decay * self._rate + (1.0 - self.spec.rate_decay) * observed

    def propose(self, _snn: SNN) -> NetworkAdjustment:
        self._proposed_current = float(
            np.clip(
                self._current + self.spec.learning_rate * (self.spec.target_rate - self._rate),
                self.spec.minimum_current,
                self.spec.maximum_current,
            )
        )
        return NetworkAdjustment()

    def commit_state(self) -> None:
        if self._proposed_current is None:
            raise RuntimeError("homeostatic drive must propose before commit")
        self._current = self._proposed_current
        self._proposed_current = None


@dataclass(frozen=True, slots=True)
class TonicDriveSpec(PopulationPluginSpec):
    """Population-local declaration for :class:`TonicDrive`."""

    current: float
    heterogeneity: float = 0.0

    def build_population_hooks(
        self, _session: Session, population: Population[Any], rng: np.random.Generator
    ) -> tuple[Hook, ...]:
        return (TonicDrive(population, self, rng),)


@dataclass(frozen=True, slots=True)
class HomeostaticDriveSpec(PopulationPluginSpec):
    """Population-local declaration for :class:`HomeostaticDrive`."""

    target_rate: float
    learning_rate: float = 0.0001
    rate_decay: float = 0.999
    minimum_current: float = -0.1
    maximum_current: float = 0.1
    initial_current: float = 0.0

    def build_population_hooks(
        self, _session: Session, population: Population[Any], _rng: np.random.Generator
    ) -> tuple[Hook, ...]:
        return (HomeostaticDrive(population, self),)
