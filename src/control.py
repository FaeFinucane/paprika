from dataclasses import dataclass, field

import numpy as np


@dataclass
class BackgroundDrive:
    population: object
    amplitude: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def produce(self):
        return np.full(self.population.bounds.stop - self.population.bounds.start, self.amplitude)


@dataclass
class Metrics:
    ticks: int = 0
    spike_count: int = 0

    def observe(self, spikes):
        self.ticks += 1
        self.spike_count += int(spikes.values.sum())


@dataclass
class Homeostasis:
    population: object
    target_rate: float = 0.1
    gain: float = 0.01
    drive: float = 0.0

    def observe(self, spikes):
        self.drive += self.gain * (
            self.target_rate - float(spikes.population(self.population).mean())
        )

    def produce(self):
        return np.full(self.population.bounds.stop - self.population.bounds.start, self.drive)


@dataclass
class Control:
    pre: list = field(default_factory=list)
    post: list = field(default_factory=list)
