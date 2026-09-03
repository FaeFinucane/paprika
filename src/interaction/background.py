
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..network.population import Population

from .plugins import Influence, Drives


@dataclass
class BackgroundDrive(Influence):
    populations: Sequence[Population[Any]]
    rng: np.random.Generator
    amplitude: float = 0.12
    # Fraction of neurons driven per tick
    sparsity: float = 0.3

    def __post_init__(self):
        if not 0 < self.sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1]")

    def produce(self) -> Drives:
        drives: dict[Population[Any], np.ndarray] = {}
        for population in self.populations:
            active = self.rng.random(population.count) < self.sparsity
            current = self.rng.random(population.count) * self.amplitude
            drives[population] = np.where(active, current, 0.0)
        return Drives(drives)
