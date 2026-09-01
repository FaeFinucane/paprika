
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..network.population import Population
from ..network.snn import Spikes

class Drive(ABC):
    population: Population

    @abstractmethod
    def produce(self) -> np.ndarray:
        raise NotImplementedError

class Observer(ABC):
    @abstractmethod
    def observe(self, spikes: Spikes) -> None:
        raise NotImplementedError

@dataclass
class BackgroundDrive(Drive):
    population: Population
    amplitude: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def produce(self) -> np.ndarray:
        return np.full(self.population.bounds.stop - self.population.bounds.start, self.amplitude)

@dataclass
class Homeostasis(Drive, Observer):
    population: Population
    target_rate: float = 0.1
    gain: float = 0.01
    drive: float = 0.0

    def observe(self, spikes: Spikes):
        self.drive += self.gain * (
            self.target_rate - float(spikes.population(self.population).mean())
        )

    def produce(self) -> np.ndarray:
        return np.full(self.population.bounds.stop - self.population.bounds.start, self.drive)