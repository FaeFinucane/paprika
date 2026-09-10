"""Generic, read-only tick trace for population activity and connection state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..interaction.plugins import CurrentDrive, Observer
from ..network.population import Population
from ..network.snn import SNN, Spikes


@dataclass(frozen=True)
class TraceSample:
    tick: int
    population_rates: dict[str, float]
    population_drive_currents: dict[str, float]
    projection_strengths: dict[str, tuple[float, float, float]]


@dataclass
class NetworkTrace(Observer):
    """Capture per-tick population rates and projection strength summaries."""

    snn: SNN
    populations: Sequence[Population[Any]] | None = None
    current_drives: Sequence[CurrentDrive] = ()
    include_projection_strengths: bool = True
    samples: list[TraceSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.populations is None:
            self.populations = self.snn.layout.populations
        for population in self.populations:
            if population.layout_fingerprint != self.snn.layout.fingerprint:
                raise ValueError("trace population belongs to another network")
        for drive in self.current_drives:
            if drive.population.layout_fingerprint != self.snn.layout.fingerprint:
                raise ValueError("trace drive belongs to another network")

    def observe(self, spikes: Spikes) -> None:
        assert self.populations is not None
        rates = {
            population.spec.name: float(np.mean(spikes.population(population)))
            for population in self.populations
        }
        drive_currents = {population.spec.name: 0.0 for population in self.populations}
        for drive in self.current_drives:
            name = drive.population.spec.name
            if name in drive_currents:
                drive_currents[name] += drive.current
        strengths: dict[str, tuple[float, float, float]] = {}
        if self.include_projection_strengths:
            for name in np.unique(self.snn.synapses.projection):
                values = self.snn.synapses.strength[self.snn.synapses.projection_mask(str(name))]
                if values.size:
                    strengths[str(name)] = (
                        float(values.min()),
                        float(values.mean()),
                        float(values.max()),
                    )
        self.samples.append(TraceSample(spikes.tick, rates, drive_currents, strengths))

    def reset(self) -> None:
        self.samples.clear()

    def mean_rate(self, population_name: str) -> float:
        if not self.samples:
            raise ValueError("trace contains no samples")
        return float(np.mean([sample.population_rates[population_name] for sample in self.samples]))
