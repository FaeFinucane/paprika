"""Network orchestration and lightweight recording."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neurons import LIFNeurons
from .synapses import SparseSynapses


@dataclass
class SpikingNetwork:
    """A recurrent network whose state advances one tick at a time."""

    neurons: LIFNeurons
    synapses: SparseSynapses
    tick: int = 0
    spike_history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.synapses.neuron_count != self.neurons.count:
            raise ValueError("neuron and synapse population sizes must match")
        self._pending_current = np.zeros(self.neurons.count, dtype=float)

    @classmethod
    def random(
        cls,
        neuron_count: int,
        seed: int = 0,
        connection_probability: float = 0.1,
    ) -> "SpikingNetwork":
        rng = np.random.default_rng(seed)
        neurons = LIFNeurons(neuron_count)
        synapses = SparseSynapses.random(
            neuron_count, connection_probability, rng
        )
        return cls(neurons, synapses)

    def step(self, external_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one tick; synaptic spikes arrive on the following tick."""

        if external_current is None:
            external = np.zeros(self.neurons.count, dtype=float)
        else:
            external = np.asarray(external_current, dtype=float)
            if external.shape != (self.neurons.count,):
                raise ValueError(
                    f"external_current must have shape ({self.neurons.count},)"
                )

        spikes = self.neurons.step(self._pending_current + external)
        self._pending_current = self.synapses.transmit(spikes)
        self.spike_history.append(spikes.copy())
        self.tick += 1
        return spikes

    def reset_state(self, clear_history: bool = True) -> None:
        self.neurons.reset_state()
        self._pending_current.fill(0.0)
        self.tick = 0
        if clear_history:
            self.spike_history.clear()
