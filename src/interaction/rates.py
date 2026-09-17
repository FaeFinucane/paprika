"""Rate-coded interaction primitives.

The encoders in this module deliberately deal in population-wide firing
probabilities.  They do not attach meaning to a positive or negative subset
of neurons; signed signals are represented by deviation from a tonic rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..network.population import Population
from ..network.snn import Spikes
from .plugins import Drives, DriveSource, Observer


@dataclass
class UnipolarRateInput(DriveSource):
    """Bernoulli rate drive for an ordinary population.

    ``write`` starts (or replaces) a drive which is emitted for exactly
    ``duration`` calls to :meth:`produce`.  ``clear`` cancels it immediately.
    """

    population: Population[Any]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    duration: int = 1

    _rate: float | None = field(default=None, init=False, repr=False)
    _remaining: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    @property
    def rate(self) -> float | None:
        return self._rate

    def write(self, rate: float, duration: int | None = None) -> None:
        """Set a rate in ``[0, 1]`` and optionally override its duration."""
        rate = float(rate)
        ticks = self.duration if duration is None else int(duration)
        if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be finite and in [0, 1]")
        if ticks <= 0:
            raise ValueError("duration must be positive")
        self._rate, self._remaining = rate, ticks

    def clear(self) -> None:
        self._rate = None
        self._remaining = 0

    reset = clear

    def produce(self) -> Drives:
        if self._rate is None or self._remaining <= 0:
            return Drives()
        values = (self.rng.random(self.population.count) < self._rate).astype(float)
        self._remaining -= 1
        if self._remaining == 0:
            self._rate = None
        return Drives({self.population: values})


@dataclass
class RatePatternInput(DriveSource):
    """Bernoulli rate drive for an arbitrary unipolar population pattern."""

    population: Population[Any]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    duration: int = 1

    _rates: np.ndarray | None = field(default=None, init=False, repr=False)
    _remaining: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    def write(self, rates: ArrayLike, duration: int | None = None) -> None:
        """Set per-neuron rates in ``[0, 1]`` for a bounded duration."""
        values = np.asarray(rates, dtype=float)
        ticks = self.duration if duration is None else int(duration)
        if values.shape != (self.population.count,) or not np.all(np.isfinite(values)):
            raise ValueError("rates must be a finite vector matching the population")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("rates must be in [0, 1]")
        if ticks <= 0:
            raise ValueError("duration must be positive")
        self._rates, self._remaining = values.copy(), ticks

    def clear(self) -> None:
        self._rates = None
        self._remaining = 0

    reset = clear

    def produce(self) -> Drives:
        if self._rates is None or self._remaining <= 0:
            return Drives()
        values = (self.rng.random(self.population.count) < self._rates).astype(float)
        self._remaining -= 1
        if self._remaining == 0:
            self._rates = None
        return Drives({self.population: values})


@dataclass
class PopulationRate(Observer):
    """Low-pass estimate of a population's mean spike probability."""

    population: Population[Any]
    decay: float = 0.95
    _value: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")

    @property
    def value(self) -> float:
        return self._value

    @property
    def rate(self) -> float:
        return self._value

    def observe(self, spikes: Spikes) -> None:
        current = float(np.mean(spikes.population(self.population)))
        self._value = self.decay * self._value + (1.0 - self.decay) * current

    def reset(self) -> None:
        self._value = 0.0

    clear = reset


@dataclass
class DopamineReadout(Observer):
    """Decode a population rate as bounded signed tonic-rate deviation."""

    source: PopulationRate
    baseline: float = 0.5
    deadzone: float = 0.02
    scale: float | None = None
    _value: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.baseline <= 1.0:
            raise ValueError("baseline must be in [0, 1]")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if self.scale is None:
            self.scale = max(self.baseline, 1.0 - self.baseline)
        if not np.isfinite(self.scale) or self.scale <= self.deadzone:
            raise ValueError("scale must be greater than deadzone")

    @property
    def value(self) -> float:
        return self._value

    def decode(self, rate: float) -> float:
        delta = float(rate) - self.baseline
        magnitude = abs(delta)
        if magnitude <= self.deadzone:
            return 0.0
        assert self.scale is not None
        return float(
            np.clip(
                np.sign(delta) * (magnitude - self.deadzone) / (self.scale - self.deadzone),
                -1.0,
                1.0,
            )
        )

    def observe(self, _spikes: Spikes) -> None:
        """Decode the source rate prepared earlier in this session tick."""
        self._value = self.decode(self.source.rate)

    def reset(self) -> None:
        self.source.reset()
        self._value = 0.0

    clear = reset
