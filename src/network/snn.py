"""Deterministic compiled leaky integrate-and-fire network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .connectivity import Connectivity, SparseSynapses
from .population import Population, PopulationLayout
import numpy as np

@dataclass(frozen=True, slots=True)
class Spikes:
    values: np.ndarray
    tick: int

    def population(self, population: Population[Any]) -> np.ndarray:
        if population.layout_fingerprint != self._fingerprint:
            raise ValueError("population belongs to another layout")
        
        return self.values[population.bounds]

    _fingerprint: str = ""

    def __post_init__(self):
        # TODO: How do we ensure values is an array of bool
        if self.values.ndim != 1 or self.values.dtype != np.bool_:
            raise ValueError("spike values must be a one-dimensional boolean array")
        if self.tick < 0:
            raise ValueError("spike tick must be non-negative")


@dataclass
class LIFNeurons:
    voltage: np.ndarray
    refractory: np.ndarray
    threshold: float = 1.0
    decay: float = 0.9
    reset: float = 0.0
    refractory_ticks: int = 0

    @classmethod
    def build(cls, count: int, threshold: float = 1.0, decay: float = 0.9, reset: float = 0.0, refractory_ticks: int = 0):
        if threshold <= 0 or not 0 <= decay <= 1 or refractory_ticks < 0:
            raise ValueError("invalid LIF parameters")
        return cls(
            np.zeros(count), np.zeros(count, dtype=int), threshold, decay, reset, refractory_ticks
        )

    def step(self, current: np.ndarray) -> np.ndarray:
        self.voltage[self.refractory > 0] = self.reset
        self.refractory[self.refractory > 0] -= 1
        live = self.refractory == 0
        self.voltage[live] = self.decay * self.voltage[live] + current[live]
        spikes = live & (self.voltage >= self.threshold)
        self.voltage[spikes] = self.reset
        self.refractory[spikes] = self.refractory_ticks
        return spikes


@dataclass
class SNN:
    layout: PopulationLayout
    neurons: LIFNeurons
    synapses: SparseSynapses
    pending_current: np.ndarray
    tick: int = 0

    @staticmethod
    def build(layout: PopulationLayout, connectivity: Connectivity):
        if layout.fingerprint != connectivity.layout_fingerprint:
            raise ValueError("layout/connectivity mismatch")
        return SNN(
            layout,
            LIFNeurons.build(layout.total_count),
            connectivity.synapses,
            np.zeros(layout.total_count),
        )

    def step(self, external_current: Mapping[Population[Any], np.ndarray] | None = None):
        current = self.pending_current.copy()
        self.pending_current.fill(0)
        for pop, value in (external_current or {}).items():
            value = np.asarray(value, dtype=float)
            if pop.layout_fingerprint != self.layout.fingerprint or value.shape != (pop.spec.count,):
                raise ValueError("invalid external current")
            current[pop.bounds] += value
        emitted = self.neurons.step(current)
        if emitted.any():
            np.add.at(
                self.pending_current,
                self.synapses.target[self.synapses.active],
                self.synapses.weight[self.synapses.active]
                * emitted[self.synapses.source[self.synapses.active]],
            )
        self.tick += 1
        return Spikes(emitted.copy(), self.tick, self.layout.fingerprint)

    def apply_weight_delta(self, delta: np.ndarray):
        delta = np.asarray(delta, dtype=float)
        if delta.shape != self.synapses.weight.shape or not np.all(np.isfinite(delta)):
            raise ValueError("invalid weight delta")

        active = self.synapses.active
        weight = self.synapses.weight

        weight[active] = np.clip(
            weight[active] + delta[active],
            self.synapses.minimum,
            self.synapses.maximum,
        )

        sign_lock = self.synapses.sign_lock
        inhibitory = active & sign_lock
        excitatory = active & ~sign_lock
        weight[inhibitory] = np.minimum(weight[inhibitory], 0.0)
        weight[excitatory] = np.maximum(weight[excitatory], 0.0)
