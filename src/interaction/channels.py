from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..network.snn import Spikes
from ..network.population import FeaturePopulationSpec, NumericPopulationSpec, Population
from .plugins import Drives, Influence, Observer

@dataclass
class FeatureInChannel(Influence):
    """An input channel for encoding features into neurons and providing them as inputs to the SNN."""

    population: Population[FeaturePopulationSpec]
    encoder: PopulationEncoder

    # How long to send each feature to the input for
    duration: int = 1

    _buffer: list[str] = field(default_factory=list[str]) # TODO: Queue?
    _current_duration: int = 0

    def write(self, feature: str):
        self._buffer.append(feature)

    def clear(self):
        """Discard any queued, not-yet-produced features."""
        self._buffer.clear()
        self._current_duration = 0

    def produce(self) -> Drives:
        if len(self._buffer) == 0:
            # TODO: Allow returning empty drives
            return Drives()

        feature = self._buffer[0]
        drive = self.encoder.encode(feature, self.population)

        self._current_duration += 1
        if self._current_duration == self.duration:
            self._current_duration = 0
            self._buffer.pop(0)

        return Drives({ self.population: drive })

@dataclass
class FeatureOutChannel(Observer):
    """Observes spikes in a population and detects which features are active"""

    population: Population[FeaturePopulationSpec]
    decoder: PopulationDecoder

    _current: EventOutput | None = None

    def observe(self, spikes: Spikes) -> None:
        next_feature = self.decoder.observe(spikes, self.population)

        if next_feature is None:
            self._current = None
            return

        feature, evidence = next_feature

        # Continuing with same feature
        if self._current is not None and self._current.feature == feature:
            # Update the feature in-place
            self._current.evidence = evidence
            self._current.duration += 1
        else:
            # New feature
            self._current = EventOutput(
                feature,
                evidence,
                spikes.tick
            )

    @property
    def current(self) -> EventOutput | None:
        return self._current

    @property
    def has_feature_changed(self) -> bool:
        return self._current is not None and self._current.duration == 0

@dataclass
class EventOutput:
    feature: str
    evidence: float
    timestamp_start: int
    duration: int = 0

@dataclass
class PopulationEncoder:
    rng: np.random.Generator
    amplitude: float = 1.0

    def __post_init__(self):
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

    def encode(self, feature: str, population: Population[FeaturePopulationSpec]) -> np.ndarray:
        out = np.zeros(population.spec.count)
        bounds = population.spec.feature_bounds(feature)
        out[bounds] = self.rng.uniform(0.0, self.amplitude, size=out[bounds].shape)
        return out


@dataclass
class PopulationDecoder:
    # The minimum number of spikes in a feature population to consider it active.
    threshold: float = 1.0

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def observe(self, spikes: Spikes, population: Population[FeaturePopulationSpec]) -> tuple[str, int] | None:
        values = spikes.population(population)
        # Split to features
        features = values.reshape((len(population.spec.features), -1))

        counts = features.sum(axis=1)
        max_feature = counts.argmax()
        max_feature_count = counts[max_feature]

        if max_feature_count >= self.threshold:
            return (population.spec.features[max_feature], max_feature_count)
        return None

@dataclass(slots=True)
class NumericChannel(Influence, Observer):
    """A bidirectional numeric channel over a population's spikes.
    """

    population: Population[NumericPopulationSpec]
    rng: np.random.Generator
    trace_decay: float = 0.95

    _read_value: float = field(default=0.0, init=False, repr=False)
    _write_value: float | None = field(default=0.0, init=False, repr=False)
    # TODO: Fix scaling mechanism
    _scale: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._scale = max(self.population.spec.positive, self.population.spec.negative)

    @property
    def value(self) -> float:
        return self._read_value

    def force(self, new_value: float):
        """Force the channel to a new normalized value (roughly [-1, 1]) by manually overwriting neurons"""
        self._write_value = new_value

    def observe(self, spikes: Spikes):
        # TODO: Have different neurons represent different strengths, similar to binary
        # but with more redundancy.
        local_spikes = spikes.population(self.population)

        raw_value = (
            local_spikes[:self.population.spec.positive].sum()
            - local_spikes[self.population.spec.positive:].sum()
        )

        # EMA, not instantaneous - a sparse population's spike count at any
        # single tick is mostly 0, discarding almost all signal exactly when
        # it matters (see conversation).
        self._read_value = self.trace_decay * self._read_value + (1.0 - self.trace_decay) * (raw_value / self._scale)

    def produce(self) -> Drives:
        if self._write_value is None:
            return Drives()

        # Convert back to a raw neuron-count-equivalent for the ratio math
        # below, which operates in units of "how many neurons to activate".
        diff = (self._write_value - self._read_value) * self._scale
        if diff == 0:
            self._write_value = None
            return Drives()

        # TODO: Algorithm is very imperfect. If neurons are already being driven, our approach overwrites some of those signals, forcing a set number to fire.
        # TODO: However at the same time, to properly work with STDP, we do want to cause neurons currently being driven to fire, so that those synapses strengthen.

        # diff is the number of neurons we want to activate. First we activate the neurons with the same sign as diff,
        # then if we need to we de-activate neurons with the opposite sign.
        positive_neuron_ratio = np.clip((
            min(abs(diff), self.population.spec.positive)
            if diff > 0 else
            np.clip(-self.population.spec.positive, abs(diff) - self.population.spec.negative, 0)
        ) / self.population.spec.positive, 0.0, 1.0)
        negative_neuron_ratio = np.clip((
            min(abs(diff), self.population.spec.negative)
            if diff < 0 else
            np.clip(-self.population.spec.negative, abs(diff) - self.population.spec.positive, 0)
        ) / self.population.spec.negative, 0.0, 1.0)

        positive_drives = self.rng.choice([0.0, 1.0], size=self.population.spec.positive, p=[1 - positive_neuron_ratio, positive_neuron_ratio])
        negative_drives = self.rng.choice([0.0, 1.0], size=self.population.spec.negative, p=[1 - negative_neuron_ratio, negative_neuron_ratio])

        self._write_value = None

        return Drives({
            self.population: np.concatenate([positive_drives, negative_drives])
        })