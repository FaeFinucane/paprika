"""Scheduled hooks around vectorised network ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..plasticity.stdp import RewardModulatedSTDP
from .population_homeostasis import PopulationHomeostasis
from .runtime_metrics import RuntimeMetrics


@dataclass
class NetworkContext:
    tick: int
    spikes: np.ndarray
    voltage: np.ndarray


class NetworkPlugin(Protocol):
    interval: int

    def after_step(self, context: NetworkContext) -> None: ...


class MetricsPlugin:
    interval = 1

    def __init__(self, metrics: RuntimeMetrics) -> None:
        self.metrics = metrics

    def after_step(self, context: NetworkContext) -> None:
        self.metrics.record(context.spikes, context.voltage)


class HomeostasisPlugin:
    def __init__(self, homeostasis: PopulationHomeostasis) -> None:
        self.homeostasis = homeostasis
        self.interval = 1

    def after_step(self, context: NetworkContext) -> None:
        self.homeostasis.observe(context.spikes)


class PlasticityPlugin:
    def __init__(self, plasticity: RewardModulatedSTDP) -> None:
        self.plasticity = plasticity
        self.interval = 1

    def after_step(self, context: NetworkContext) -> None:
        self.plasticity.observe(context.spikes)
