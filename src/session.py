from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .interaction.plugins import Drive, Observer
from .network.population import Population
from .conversation import Conversation
from .network.snn import SNN, Spikes

@dataclass
class Control:
    pre: list[Drive] = field(default_factory=list[Drive])
    post: list[Observer] = field(default_factory=list[Observer])

@dataclass
class Session:
    snn: SNN
    conversation: Conversation = field(default_factory=Conversation)
    pre: list[Drive] = field(default_factory=list[Drive])
    post: list[Observer] = field(default_factory=list[Observer])

    def tick(self, drives: Mapping[Population, np.ndarray] | None = None):
        merged = {
            pop: np.asarray(value, dtype=float).copy() for pop, value in (drives or {}).items()
        }

        for service in self.pre:
            value = np.asarray(service.produce(), dtype=float)
            if value.shape != (service.population.count,):
                raise ValueError("control drive has the wrong shape")
            merged[service.population] = (
                merged.get(service.population, np.zeros(service.population.count)) + value
            )

        for pop, value in merged.items():
            if value.shape != (pop.count,):
                raise ValueError("drive has the wrong shape")
        spikes = self.snn.step(merged)

        for service in self.post:
            service.observe(spikes)
        return spikes

@dataclass
class Metrics(Observer):
    ticks: int = 0
    spike_count: int = 0

    def observe(self, spikes: Spikes):
        self.ticks += 1
        self.spike_count += int(spikes.values.sum())