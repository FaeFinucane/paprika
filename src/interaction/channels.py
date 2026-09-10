from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..network.population import FeaturePopulationSpec, Population
from ..network.snn import Spikes
from .plugins import Drives, DriveSource, Observer


@dataclass
class FeatureInChannel(DriveSource):
    """An input channel for encoding features into neurons and providing them as inputs to the SNN."""

    population: Population[FeaturePopulationSpec]
    encoder: PopulationEncoder

    # How long to send each feature to the input for
    duration: int = 1

    _buffer: list[str] = field(default_factory=list[str])  # TODO: Queue?
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

        return Drives({self.population: drive})


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
            self._current = EventOutput(feature, evidence, spikes.tick)

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

    def observe(
        self, spikes: Spikes, population: Population[FeaturePopulationSpec]
    ) -> tuple[str, int] | None:
        values = spikes.population(population)
        # Split to features
        features = values.reshape((len(population.spec.features), -1))

        counts = features.sum(axis=1)
        max_feature = counts.argmax()
        max_feature_count = counts[max_feature]

        if max_feature_count >= self.threshold:
            return (population.spec.features[max_feature], max_feature_count)
        return None
