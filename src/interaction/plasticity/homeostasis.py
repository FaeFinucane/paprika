"""Slow, reward-independent plasticity that regulates network activity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...network.adjustments import NetworkAdjustment
from ...network.population import Population
from ...network.snn import SNN, Spikes
from ..plugins import SpikeAdaptation


@dataclass
class SynapticScaling(SpikeAdaptation):
    """Very slow multiplicative scaling of explicitly scalable E inputs."""

    snn: SNN
    population: Population[Any]
    target_rate: float
    learning_rate: float = 0.00001
    rate_decay: float = 0.9999
    _rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.target_rate <= 1 or self.learning_rate < 0 or not 0 <= self.rate_decay < 1:
            raise ValueError("invalid synaptic-scaling configuration")
        if self.population.layout_fingerprint != self.snn.layout.fingerprint:
            raise ValueError("population belongs to another network")

    @property
    def rate(self) -> float:
        return self._rate

    def observe(self, spikes: Spikes) -> None:
        observed = float(np.mean(spikes.population(self.population)))
        self._rate = self.rate_decay * self._rate + (1.0 - self.rate_decay) * observed

    def propose(self, snn: SNN) -> NetworkAdjustment:
        if snn is not self.snn:
            raise ValueError("synaptic scaling belongs to another network")
        synapses = self.snn.synapses
        source_is_excitatory = self.snn.layout.output_kinds[synapses.source] == "excitatory"
        target = (synapses.target >= self.population.bounds.start) & (
            synapses.target < self.population.bounds.stop
        )
        mask = synapses.active & synapses.scalable & source_is_excitatory & target
        if not mask.any():
            return NetworkAdjustment()
        factor = float(np.exp(self.learning_rate * (self.target_rate - self._rate)))
        delta = np.zeros_like(synapses.strength)
        delta[mask] = synapses.strength[mask] * (factor - 1.0)
        return NetworkAdjustment(delta, mask)
