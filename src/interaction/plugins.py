
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

from ..network.population import Population
from ..network.snn import Spikes

@dataclass
class Drives:
    drives: Mapping[Population[Any], np.ndarray] = field(default_factory=dict[Population[Any], np.ndarray])

    def accumulate(self, other: Drives) -> Drives:
        merged = dict(self.drives)
        for pop, value in other.drives.items():
            if pop in merged:
                merged[pop] = np.clip(merged[pop] + value, 0, 1)  # TODO: More advanced clamping logic?
            else:
                merged[pop] = value
        return Drives(merged)

class Influence(ABC):
    @abstractmethod
    def produce(self) -> Drives:
        raise NotImplementedError

class Observer(ABC):
    @abstractmethod
    def observe(self, spikes: Spikes) -> None:
        raise NotImplementedError
