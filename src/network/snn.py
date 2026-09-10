"""Compiled leaky integrate-and-fire network runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .adjustments import NetworkAdjustment
from .connectivity import SparseSynapses
from .population import Population, PopulationLayout


@dataclass(frozen=True, slots=True)
class Spikes:
    values: np.ndarray
    tick: int
    _fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.dtype != np.bool_:
            raise ValueError("spikes must be a one-dimensional boolean array")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")

    def population(self, population: Population[Any]) -> np.ndarray:
        if population.layout_fingerprint != self._fingerprint:
            raise ValueError("population belongs to another layout")
        return self.values[population.bounds]


@dataclass(frozen=True, slots=True)
class LIFDynamics:
    """Shared, fixed dynamics for every ordinary neuron in the simulation."""

    threshold: float = 1.0
    decay: float = 0.9
    reset: float = -0.5
    refractory_ticks: int = 0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.threshold)
            or self.threshold <= 0
            or not np.isfinite(self.decay)
            or not 0 <= self.decay <= 1
            or not np.isfinite(self.reset)
            or self.refractory_ticks < 0
        ):
            raise ValueError("invalid shared neuron dynamics")


DEFAULT_LIF_DYNAMICS = LIFDynamics()


@dataclass
class LIFNeurons:
    count: int
    dynamics: LIFDynamics = DEFAULT_LIF_DYNAMICS
    voltage: np.ndarray = field(init=False)
    refractory: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("neuron count must be positive")
        self.voltage = np.zeros(self.count, dtype=float)
        self.refractory = np.zeros(self.count, dtype=int)

    def step(self, current: np.ndarray) -> np.ndarray:
        if current.shape != self.voltage.shape or not np.all(np.isfinite(current)):
            raise ValueError("invalid current")
        refractory = self.refractory > 0
        self.voltage[refractory] = self.dynamics.reset
        self.refractory[refractory] -= 1
        live = self.refractory == 0
        self.voltage[live] = self.dynamics.decay * self.voltage[live] + current[live]
        spikes = live & (self.voltage >= self.dynamics.threshold)
        self.voltage[spikes] = self.dynamics.reset
        self.refractory[spikes] = self.dynamics.refractory_ticks
        return spikes


@dataclass
class SNN:
    layout: PopulationLayout
    neurons: LIFNeurons
    synapses: SparseSynapses
    next_current: np.ndarray
    tick: int = 0

    @staticmethod
    def build(layout: PopulationLayout, synapses: SparseSynapses) -> "SNN":
        return SNN(
            layout,
            LIFNeurons(layout.total_count),
            synapses,
            np.zeros(layout.total_count, dtype=float),
        )

    def step(self, external_current: Mapping[Population[Any], np.ndarray] | None = None) -> Spikes:
        current = self.next_current.copy()
        self.next_current.fill(0.0)
        for population, value in (external_current or {}).items():
            value = np.asarray(value, dtype=float)
            if (
                population.layout_fingerprint != self.layout.fingerprint
                or value.shape != (population.count,)
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("invalid external current")
            current[population.bounds] += value

        emitted = self.neurons.step(current)
        active = self.synapses.active
        if emitted.any() and active.any():
            transmitted = active & emitted[self.synapses.source]
            if transmitted.any():
                signed = self.layout.output_sign[self.synapses.source[transmitted]]
                if np.any(signed == 0):
                    raise RuntimeError("modulatory neuron has an ordinary projection")
                np.add.at(
                    self.next_current,
                    self.synapses.target[transmitted],
                    signed * self.synapses.strength[transmitted],
                )
        self.tick += 1
        return Spikes(emitted.copy(), self.tick, self.layout.fingerprint)

    def commit(self, adjustments: Sequence[NetworkAdjustment]) -> None:
        """Validate and atomically apply accumulated parameter adjustments."""
        strength_delta = np.zeros_like(self.synapses.strength)
        strength_mask = np.zeros_like(self.synapses.active)
        for adjustment in adjustments:
            if adjustment.strength_delta is not None:
                assert adjustment.strength_mask is not None
                delta, mask = self.synapses.validate_strength_delta(
                    adjustment.strength_delta, adjustment.strength_mask
                )
                strength_delta[mask] += delta[mask]
                strength_mask |= mask
        if strength_mask.any():
            self.synapses.apply_strength_delta(strength_delta, strength_mask)
