from dataclasses import dataclass
from typing import Sequence

from ..network.population import Population
from ..network.snn import Spikes
from .plugins import Drives, Influence, Observer

@dataclass
class Homeostasis(Influence, Observer):
    populations: Sequence[Population]
    target_rate: float = 0.1
    gain: float = 0.01
    drive: float = 0.0

    def observe(self, spikes: Spikes):
        # TODO:
        return

    def produce(self) -> Drives:
        return Drives() # TODO: