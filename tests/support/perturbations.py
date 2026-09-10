"""Deliberately crude, test-only mutations of transient network state.

These helpers preserve network definition, synaptic strengths, and intrinsic
parameters.  They let a system test start the normal dynamics from an unusual
state without treating the perturbation as an ordinary circuit input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from src.network.population import Population
from src.network.snn import SNN


def _indices(snn: SNN, population: Population[Any], count: int | None = None) -> np.ndarray:
    if population.layout_fingerprint != snn.layout.fingerprint:
        raise ValueError("population belongs to another network")
    if count is not None and not 0 < count <= population.count:
        raise ValueError("count must be between one and the population size")
    stop = population.bounds.start + (count or population.count)
    return np.arange(population.bounds.start, stop)


def set_membrane_voltage(
    snn: SNN, population: Population[Any], voltage: float, *, count: int | None = None
) -> None:
    """Set selected neurons to one explicit membrane-voltage state."""
    if not np.isfinite(voltage):
        raise ValueError("voltage must be finite")
    snn.neurons.voltage[_indices(snn, population, count)] = voltage


def add_membrane_voltage(
    snn: SNN, population: Population[Any], delta: float, *, count: int | None = None
) -> None:
    """Depolarize or hyperpolarize selected neurons without changing parameters."""
    if not np.isfinite(delta):
        raise ValueError("voltage delta must be finite")
    snn.neurons.voltage[_indices(snn, population, count)] += delta


def silence_neurons(snn: SNN, population: Population[Any], *, count: int | None = None) -> None:
    """Reset selected neurons and discard already scheduled current into them."""
    indices = _indices(snn, population, count)
    snn.neurons.voltage[indices] = snn.neurons.dynamics.reset
    snn.neurons.refractory[indices] = 0
    snn.next_current[indices] = 0.0


def clear_transient_activity(snn: SNN) -> None:
    """Return all neurons to reset state without altering learned network state."""
    snn.neurons.voltage.fill(snn.neurons.dynamics.reset)
    snn.neurons.refractory.fill(0)
    snn.next_current.fill(0.0)
