"""Vectorised simulation core with explicit one-tick synaptic delay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neurons import LIFNeurons
from .synapses import SparseSynapses


@dataclass
class NetworkCore:
    """Own LIF state, sparse connectivity, weights, and pending current."""

    neurons: LIFNeurons
    synapses: SparseSynapses
    tick: int = 0

    def __post_init__(self) -> None:
        if self.synapses.neuron_count != self.neurons.count:
            raise ValueError("neuron and synapse population sizes must match")
        self.pending_current = np.zeros(self.neurons.count, dtype=float)
        self._input = np.zeros(self.neurons.count, dtype=float)
        self._spike_buffers = (
            np.zeros(self.neurons.count, dtype=bool),
            np.zeros(self.neurons.count, dtype=bool),
        )
        self._spike_buffer_index = 0
        self.synapses.freeze_connectivity()

    @classmethod
    def random(
        cls, neuron_count: int, seed: int = 0, connection_probability: float = 0.1
    ) -> "NetworkCore":
        rng = np.random.default_rng(seed)
        return cls(
            LIFNeurons(neuron_count),
            SparseSynapses.random(neuron_count, connection_probability, rng),
        )

    def step(self, external_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one tick; spikes emitted now arrive on the following tick."""
        if external_current is None:
            self._input[:] = self.pending_current
        else:
            external = np.asarray(external_current, dtype=float)
            if external.shape != (self.neurons.count,):
                raise ValueError(f"external_current must have shape ({self.neurons.count},)")
            np.add(self.pending_current, external, out=self._input)
        self._spike_buffer_index ^= 1
        spikes = self._spike_buffers[self._spike_buffer_index]
        spikes[:] = self.neurons.step(self._input)
        self.synapses.transmit_into(spikes, self.pending_current)
        self.tick += 1
        return spikes

    def reset_state(self) -> None:
        self.neurons.reset_state()
        self.pending_current.fill(0.0)
        self.tick = 0
        for buffer in self._spike_buffers:
            buffer.fill(False)

    def reset_synaptic_activity(self) -> None:
        self.pending_current.fill(0.0)

    def reset_neuron_state(self) -> None:
        self.neurons.reset_state()

    def state_snapshot(self) -> dict[str, np.ndarray | int]:
        return {
            "voltage": self.neurons.voltage.copy(),
            "refractory": self.neurons.refractory.copy(),
            "pending_current": self.pending_current.copy(),
            "spike_buffers": np.stack(self._spike_buffers).copy(),
            "spike_buffer_index": self._spike_buffer_index,
            "tick": self.tick,
        }

    def restore_state(self, state: dict[str, np.ndarray | int]) -> None:
        voltage = np.asarray(state["voltage"], dtype=float)
        refractory = np.asarray(state["refractory"], dtype=np.int64)
        pending = np.asarray(state["pending_current"], dtype=float)
        expected = (self.neurons.count,)
        if voltage.shape != expected or refractory.shape != expected or pending.shape != expected:
            raise ValueError("network state arrays have the wrong shape")
        if not np.isfinite(voltage).all() or not np.isfinite(pending).all():
            raise ValueError("network state arrays must be finite")
        if np.any(refractory < 0) or int(state["tick"]) < 0:
            raise ValueError("network state counters must be non-negative")
        self.neurons.voltage[:] = voltage
        self.neurons.refractory[:] = refractory
        self.pending_current[:] = pending
        buffers = np.asarray(
            state.get("spike_buffers", np.zeros((2, self.neurons.count), dtype=bool)), dtype=bool
        )
        if buffers.shape != (2, self.neurons.count):
            raise ValueError("network spike buffers have the wrong shape")
        for target, source in zip(self._spike_buffers, buffers):
            target[:] = source
        self._spike_buffer_index = int(state.get("spike_buffer_index", 0))
        if self._spike_buffer_index not in (0, 1):
            raise ValueError("network spike buffer index must be 0 or 1")
        self.tick = int(state["tick"])

    def copy(self) -> "NetworkCore":
        copied = NetworkCore(
            LIFNeurons(
                self.neurons.count,
                self.neurons.dt,
                self.neurons.tau_membrane,
                self.neurons.resting_potential,
                self.neurons.reset_potential,
                self.neurons.threshold,
                self.neurons.refractory_ticks,
            ),
            SparseSynapses(
                self.synapses.source.copy(),
                self.synapses.target.copy(),
                self.synapses.weight.copy(),
                self.synapses.neuron_count,
            ),
            self.tick,
        )
        copied.neurons.voltage[:] = self.neurons.voltage
        copied.neurons.refractory[:] = self.neurons.refractory
        copied.pending_current[:] = self.pending_current
        for target, source in zip(copied._spike_buffers, self._spike_buffers):
            target[:] = source
        copied._spike_buffer_index = self._spike_buffer_index
        return copied
