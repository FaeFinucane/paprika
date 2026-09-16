from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from .interaction.plugins import (
    Drives,
    DriveSource,
    Hook,
    Observer,
    SpikeAdaptation,
    StatefulAdaptation,
)
from .network.snn import SNN


@dataclass
class Session:
    snn: SNN
    drive_sources: list[DriveSource] = field(default_factory=list)
    observers: list[Observer] = field(default_factory=list)
    adaptations: list[SpikeAdaptation] = field(default_factory=list)
    stateful_adaptations: list[StatefulAdaptation] = field(default_factory=list)
    adaptations_enabled: bool = True

    @classmethod
    def build(cls, snn: SNN, hooks: Sequence[Hook]) -> "Session":
        drive_sources: list[DriveSource] = []
        observers: list[Observer] = []
        adaptations: list[SpikeAdaptation] = []
        stateful_adaptations: list[StatefulAdaptation] = []
        for hook in hooks:
            registered = False
            if isinstance(hook, DriveSource):
                drive_sources.append(hook)
                registered = True
            if isinstance(hook, Observer):
                observers.append(hook)
                registered = True
            if isinstance(hook, SpikeAdaptation):
                adaptations.append(hook)
            if isinstance(hook, StatefulAdaptation):
                stateful_adaptations.append(hook)
            if not registered:
                raise TypeError("session hook does not implement a lifecycle role")
        return cls(snn, drive_sources, observers, adaptations, stateful_adaptations)

    @contextmanager
    def frozen_adaptations(self):
        """Temporarily run the circuit without any adaptation-side mutation."""
        previous = self.adaptations_enabled
        self.adaptations_enabled = False
        try:
            yield self
        finally:
            self.adaptations_enabled = previous

    def tick(self):
        drives = Drives()

        for source in self.drive_sources:
            drives = drives.accumulate(source.produce())

        spikes = self.snn.step(drives.drives)

        for observer in self.observers:
            if not self.adaptations_enabled and isinstance(observer, SpikeAdaptation):
                continue
            observer.observe(spikes)

        if self.adaptations_enabled:
            self.snn.commit([adaptation.propose(self.snn) for adaptation in self.adaptations])
            for adaptation in self.stateful_adaptations:
                adaptation.commit_state()

        return spikes
