"""Explicit ongoing-current sources for otherwise input-driven networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..network.adjustments import NetworkAdjustment
from ..network.population import Population
from ..network.snn import SNN, Spikes
from .plugins import Drives, DriveSource, StatefulAdaptation


@dataclass
class TonicDrive(DriveSource):
    """A named constant current source for one population."""

    population: Population[Any]
    current: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.current):
            raise ValueError("tonic current must be finite")

    def produce(self) -> Drives:
        return Drives({self.population: np.full(self.population.count, self.current)})


@dataclass
class HomeostaticDrive(DriveSource, StatefulAdaptation):
    """Slow, bounded rate regulation through an explicit current source.

    The controller owns its current rather than changing hidden neuron
    parameters. Its output is consequently visible in circuit construction and
    can be removed from a session to test the recurrent circuit on its own.
    """

    population: Population[Any]
    target_rate: float
    learning_rate: float = 0.0001
    rate_decay: float = 0.999
    minimum_current: float = -0.1
    maximum_current: float = 0.1
    initial_current: float = 0.0
    enabled: bool = True
    _rate: float = field(default=0.0, init=False, repr=False)
    _current: float = field(init=False, repr=False)
    _proposed_current: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not 0 <= self.target_rate <= 1
            or self.learning_rate < 0
            or not 0 <= self.rate_decay < 1
            or not np.isfinite(self.minimum_current)
            or not np.isfinite(self.maximum_current)
            or self.minimum_current > self.maximum_current
            or not self.minimum_current <= self.initial_current <= self.maximum_current
        ):
            raise ValueError("invalid homeostatic-drive configuration")
        self._current = self.initial_current

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def current(self) -> float:
        """Current emitted on the next tick; zero when the drive is disabled."""
        return self._current if self.enabled else 0.0

    def produce(self) -> Drives:
        if not self.enabled:
            return Drives()
        return Drives({self.population: np.full(self.population.count, self._current)})

    def observe(self, spikes: Spikes) -> None:
        observed = float(np.mean(spikes.population(self.population)))
        self._rate = self.rate_decay * self._rate + (1.0 - self.rate_decay) * observed

    def propose(self, _snn: SNN) -> NetworkAdjustment:
        self._proposed_current = float(
            np.clip(
                self._current + self.learning_rate * (self.target_rate - self._rate),
                self.minimum_current,
                self.maximum_current,
            )
        )
        return NetworkAdjustment()

    def commit_state(self) -> None:
        if self._proposed_current is None:
            raise RuntimeError("homeostatic drive must propose before commit")
        self._current = self._proposed_current
        self._proposed_current = None
