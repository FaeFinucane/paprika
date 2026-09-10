from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np

from ..network.adjustments import NetworkAdjustment
from ..network.population import Population
from ..network.snn import SNN, Spikes


@dataclass
class Drives:
    drives: Mapping[Population[Any], np.ndarray] = field(
        default_factory=dict[Population[Any], np.ndarray]
    )

    def accumulate(self, other: Drives) -> Drives:
        merged = dict(self.drives)
        for pop, value in other.drives.items():
            if pop in merged:
                # Current is signed: rate encoders produce non-negative drive,
                # while a comparator or inhibitory influence may legitimately
                # supply a negative contribution.
                merged[pop] = merged[pop] + value
            else:
                merged[pop] = value
        return Drives(merged)


@runtime_checkable
class DriveSource(Protocol):
    """Produces external current before a network tick."""

    def produce(self) -> Drives:
        ...


@runtime_checkable
class CurrentDrive(DriveSource, Protocol):
    """A drive whose uniform current can be inspected by diagnostics."""

    population: Population[Any]

    @property
    def current(self) -> float:
        ...


@runtime_checkable
class Observer(Protocol):
    """Observes the spikes emitted by a completed network tick."""

    def observe(self, spikes: Spikes) -> None:
        ...


@runtime_checkable
class SpikeAdaptation(Observer, Protocol):
    """Observes spikes, then proposes future parameter changes."""

    def propose(self, snn: SNN) -> NetworkAdjustment:
        ...


@runtime_checkable
class StatefulAdaptation(SpikeAdaptation, Protocol):
    """An adaptation with plugin-owned state to finalize at tick commit."""

    def commit_state(self) -> None:
        ...


Hook: TypeAlias = DriveSource | Observer | SpikeAdaptation
