"""Cheap structural and numerical health checks for a running network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..network.snn import SNN


@dataclass(frozen=True)
class NetworkHealth:
    valid: bool
    issues: tuple[str, ...]


def inspect_network(snn: SNN) -> NetworkHealth:
    """Return invariant violations without mutating network state."""
    synapses = snn.synapses
    issues: list[str] = []
    if not np.all(np.isfinite(synapses.strength)):
        issues.append("non-finite synaptic strength")
    if np.any(synapses.strength < synapses.minimum) or np.any(synapses.strength > synapses.maximum):
        issues.append("synaptic strength outside bounds")
    if np.any(synapses.strength < 0):
        issues.append("negative stored synaptic strength")
    if not np.all(np.isfinite(snn.neurons.voltage)):
        issues.append("non-finite neuron voltage")
    if not np.all(np.isfinite(snn.next_current)):
        issues.append("non-finite scheduled current")
    return NetworkHealth(not issues, tuple(issues))
