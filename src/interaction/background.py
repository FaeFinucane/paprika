
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..network.population import Population

from .plugins import Influence, Drives


@dataclass
class BackgroundDrive(Influence):
    populations: Sequence[Population[Any]]
    amplitude: float
    rng: np.random.Generator

    def produce(self) -> Drives:
        return Drives({
            population: self.rng.random(population.count) * self.amplitude for population in self.populations
        })
