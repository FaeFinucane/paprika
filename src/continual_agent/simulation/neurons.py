"""Neuron population models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LIFNeurons:
    """A vectorised population of leaky integrate-and-fire neurons.

    Currents and voltages use arbitrary normalised units. A call to ``step``
    advances the entire population by one fixed simulation tick.
    """

    count: int
    dt: float = 1.0
    tau_membrane: float = 10.0
    resting_potential: float = 0.0
    reset_potential: float = 0.0
    threshold: float = 1.0
    refractory_ticks: int = 2

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.dt <= 0 or self.tau_membrane <= 0:
            raise ValueError("dt and tau_membrane must be positive")
        if self.refractory_ticks < 0:
            raise ValueError("refractory_ticks cannot be negative")

        self.voltage = np.full(self.count, self.resting_potential, dtype=float)
        self.refractory = np.zeros(self.count, dtype=np.int64)

    def reset_state(self) -> None:
        """Return the population to its resting state."""

        self.voltage.fill(self.resting_potential)
        self.refractory.fill(0)

    def step(self, input_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one tick and return a boolean vector of emitted spikes."""

        if input_current is None:
            current = np.zeros(self.count, dtype=float)
        else:
            current = np.asarray(input_current, dtype=float)
            if current.shape != (self.count,):
                raise ValueError(f"input_current must have shape ({self.count},)")

        active = self.refractory == 0
        self.refractory[~active] -= 1
        self.voltage[active] += (self.dt / self.tau_membrane) * (
            -(self.voltage[active] - self.resting_potential) + current[active]
        )

        spikes = active & (self.voltage >= self.threshold)
        self.voltage[spikes] = self.reset_potential
        self.refractory[spikes] = self.refractory_ticks
        return spikes
