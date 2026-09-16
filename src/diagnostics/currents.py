"""Read-only attribution of synaptic current scheduled by a spike event."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..network.population import Population
from ..network.snn import SNN, Spikes


def incoming_projection_currents(
    snn: SNN, spikes: Spikes, target: Population[Any]
) -> dict[str, float]:
    """Return mean next-tick current into ``target``, grouped by projection.

    This attributes only ordinary synaptic transmission scheduled by ``spikes``;
    explicit drives are intentionally excluded.
    """
    if target.layout_fingerprint != snn.layout.fingerprint:
        raise ValueError("target population belongs to another network")
    if spikes._fingerprint != snn.layout.fingerprint:
        raise ValueError("spikes belong to another network")
    synapses = snn.synapses
    incoming = (synapses.target >= target.bounds.start) & (synapses.target < target.bounds.stop)
    transmitted = synapses.active & spikes.values[synapses.source] & incoming
    currents: dict[str, float] = {}
    for projection in np.unique(synapses.projection[transmitted]):
        mask = transmitted & (synapses.projection == projection)
        signs = snn.layout.output_sign[synapses.source[mask]]
        currents[str(projection)] = float(np.sum(signs * synapses.strength[mask]) / target.count)
    return currents
