from dataclasses import dataclass, field

from .interaction.plugins import Drives, Influence, Observer
from .network.snn import SNN, Spikes

@dataclass
class Control:
    pre: list[Influence] = field(default_factory=list[Influence])
    post: list[Observer] = field(default_factory=list[Observer])

@dataclass
class Session:
    snn: SNN
    pre: list[Influence] = field(default_factory=list[Influence])
    post: list[Observer] = field(default_factory=list[Observer])

    def tick(self):
        drives = Drives()

        for influence in self.pre:
            drives.accumulate(influence.produce())

        spikes = self.snn.step(drives.drives)

        for influence in self.post:
            influence.observe(spikes)
            
        return spikes

@dataclass
class Metrics(Observer):
    ticks: int = 0
    spike_count: int = 0

    def observe(self, spikes: Spikes):
        self.ticks += 1
        self.spike_count += int(spikes.values.sum())