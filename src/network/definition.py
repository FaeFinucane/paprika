"""Declarative, reusable setup for ordinary spiking networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectivity import ConnectionSpec, build_synapses
from .population import PopulationLayout, PopulationSpec
from .snn import SNN


@dataclass(frozen=True)
class NetworkDefinition:
    """Population and projection declarations that compile into one SNN.

    This is deliberately generic: task architectures add their own inputs,
    observers, and learning rules after obtaining the compiled network.
    """

    populations: tuple[PopulationSpec, ...]
    connections: tuple[ConnectionSpec, ...]

    def compile(self, rng: np.random.Generator) -> SNN:
        layout = PopulationLayout.build(self.populations)
        return SNN.build(layout, build_synapses(layout, self.connections, rng))
