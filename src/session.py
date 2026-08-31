from dataclasses import dataclass, field

import numpy as np

from .conversation import Conversation
from .network import SNN


@dataclass
class Session:
    snn: SNN
    conversation: Conversation = field(default_factory=Conversation)
    pre: list = field(default_factory=list)
    post: list = field(default_factory=list)

    def tick(self, drives=None):
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
