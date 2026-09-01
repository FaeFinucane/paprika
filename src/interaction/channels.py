from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..network.snn import Spikes
from ..network.population import FeaturePopulation


@dataclass
class PopulationEncoder:
    amplitude: float = 1.0

    def encode(self, features: tuple[str, ...], population: FeaturePopulation) -> np.ndarray:
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        out = np.zeros(population.count)
        # TODO: Could use the 'step' value in slice?
        for feature in features:
            out[
                population.feature_bounds(feature).start
                - population.bounds.start : population.feature_bounds(feature).stop
                - population.bounds.start
            ] = self.amplitude
        return out


@dataclass(frozen=True)
class FeatureObservation:
    feature: str
    evidence: float
    active: bool
    timestamp: int


@dataclass
class PopulationDetector:
    threshold: float = 1.0

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def observe(self, spikes: Spikes, population: FeaturePopulation):
        values = spikes.population(population)
        # TODO: Should return ONE feature observation, not multiple.
        return tuple(
            FeatureObservation(
                f,
                float(
                    values[
                        s.start - population.bounds.start : s.stop - population.bounds.start
                    ].sum()
                ),
                bool(
                    values[
                        s.start - population.bounds.start : s.stop - population.bounds.start
                    ].sum()
                    >= self.threshold
                ),
                spikes.tick,
            )
            for f, s in population.features.items()
        )


@dataclass
class InputChannel:
    population: FeaturePopulation
    encoder: PopulationEncoder

    def encode(self, features: tuple[str, ...]) -> np.ndarray:
        return self.encoder.encode(features, self.population)


@dataclass
class OutputChannel:
    population: FeaturePopulation
    detector: PopulationDetector

    def observe(self, spikes: Spikes):
        return self.detector.observe(spikes, self.population)


@dataclass(frozen=True)
class EventInput:
    channel: InputChannel
    feature: str
    duration: int = 1

    def __post_init__(self):
        self.channel.population.feature_bounds(self.feature)
        if self.duration <= 0:
            raise ValueError("duration must be positive")


# TODO: EventOutput takes PopulationDetector to determine what the output is. I think.
# But really the OutputChannel should just produce an EventOutput
@dataclass
class EventOutput:
    channel: OutputChannel
    features: tuple[str, ...] | None = None
    eos: str | None = None

    def __post_init__(self):
        if self.features is not None:
            unknown = set(self.features) - set(self.channel.population.features)
            if unknown:
                raise ValueError(f"unknown output features: {sorted(unknown)!r}")
        if self.eos is not None and self.eos not in self.channel.population.features:
            raise ValueError("EOS feature is not in the output population")

    def consume(self, observations: tuple[FeatureObservation, ...]) -> FeatureObservation | None:
        allowed = set(self.features) if self.features is not None else None
        candidates = [
            x for x in observations if x.active and (allowed is None or x.feature in allowed)
        ]
        return max(candidates, key=lambda x: (x.evidence, x.feature)) if candidates else None
