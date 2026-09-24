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
    _hooks: list[Hook] = field(default_factory=list, init=False, repr=False)
    _drive_sources: list[DriveSource] = field(default_factory=list, init=False, repr=False)
    _observers: list[Observer] = field(default_factory=list, init=False, repr=False)
    _adaptations: list[SpikeAdaptation] = field(default_factory=list, init=False, repr=False)
    _stateful_adaptations: list[StatefulAdaptation] = field(
        default_factory=list, init=False, repr=False
    )
    adaptations_enabled: bool = True

    @classmethod
    def build(cls, snn: SNN, hooks: Sequence[Hook]) -> "Session":
        session = cls(snn)
        session.add(*hooks)
        return session

    @property
    def drive_sources(self) -> tuple[DriveSource, ...]:
        return tuple(self._drive_sources)

    @property
    def observers(self) -> tuple[Observer, ...]:
        return tuple(self._observers)

    @property
    def adaptations(self) -> tuple[SpikeAdaptation, ...]:
        return tuple(self._adaptations)

    @property
    def stateful_adaptations(self) -> tuple[StatefulAdaptation, ...]:
        return tuple(self._stateful_adaptations)

    def add(self, *hooks: Hook) -> None:
        """Register hooks through every lifecycle role they implement.

        A hook that both observes spikes and proposes updates is registered for
        both phases atomically. Runtime code must use this method rather than
        mutating lifecycle lists, so a plasticity rule cannot be scheduled for
        commits while silently missing its observation phase.
        """
        for hook in hooks:
            if any(hook is registered for registered in self._hooks):
                raise ValueError("session hook is already registered")
            registered = False
            if isinstance(hook, DriveSource):
                self._drive_sources.append(hook)
                registered = True
            if isinstance(hook, Observer):
                self._observers.append(hook)
                registered = True
            if isinstance(hook, SpikeAdaptation):
                self._adaptations.append(hook)
            if isinstance(hook, StatefulAdaptation):
                self._stateful_adaptations.append(hook)
            if not registered:
                raise TypeError("session hook does not implement a lifecycle role")
            self._hooks.append(hook)

    def remove(self, *hooks: Hook) -> None:
        """Unregister hooks from every lifecycle role they implement.

        This is intentionally identity-based, mirroring ``add``. It supports
        temporary experimental ablations without adding behaviour flags to the
        circuit or plasticity implementation.
        """
        for hook in hooks:
            if not any(hook is registered for registered in self._hooks):
                raise ValueError("session hook is not registered")
            self._hooks = [registered for registered in self._hooks if registered is not hook]
            self._drive_sources = [
                registered for registered in self._drive_sources if registered is not hook
            ]
            self._observers = [
                registered for registered in self._observers if registered is not hook
            ]
            self._adaptations = [
                registered for registered in self._adaptations if registered is not hook
            ]
            self._stateful_adaptations = [
                registered for registered in self._stateful_adaptations if registered is not hook
            ]

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

        for source in self._drive_sources:
            drives = drives.accumulate(source.produce())

        spikes = self.snn.step(drives.drives)

        for observer in self._observers:
            if not self.adaptations_enabled and isinstance(observer, SpikeAdaptation):
                continue
            observer.observe(spikes)

        if self.adaptations_enabled:
            self.snn.commit([adaptation.propose(self.snn) for adaptation in self._adaptations])
            for adaptation in self._stateful_adaptations:
                adaptation.commit_state()

        return spikes
