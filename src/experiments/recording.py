"""Small, composable sources for experiment reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..interaction.plugins import Observer, ValueObserver
from ..network.connectivity import ConnectionSpec
from ..network.snn import SNN, Spikes
from ..session import Session

SignalValue = float | np.ndarray


def _copy_value(value: SignalValue) -> SignalValue:
    if isinstance(value, np.ndarray):
        return value.copy()
    return float(value)


@dataclass
class SignalRecorder(Observer):
    """Record the current values of selected streaming observers each tick.

    Use :meth:`from_session` after all signal observers have been registered,
    then add the recorder to the session. Session observer order guarantees
    that each source has already updated before this recorder samples it.
    """

    signals: tuple[ValueObserver, ...]
    _samples: dict[str, list[SignalValue]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        names = [signal.report_name for signal in self.signals]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate report names: {', '.join(duplicates)}")
        self._samples = {name: [] for name in names}

    @classmethod
    def from_session(cls, session: Session) -> SignalRecorder:
        """Discover all streaming observers already registered with ``session``."""
        return cls(
            tuple(observer for observer in session.observers if isinstance(observer, ValueObserver))
        )

    @property
    def sample_count(self) -> int:
        return len(next(iter(self._samples.values()), ()))

    @property
    def report_names(self) -> tuple[str, ...]:
        return tuple(self._samples)

    def observe(self, _spikes: Spikes) -> None:
        for signal in self.signals:
            self._samples[signal.report_name].append(_copy_value(signal.value))

    def samples_since(self, start: int) -> dict[str, np.ndarray]:
        """Return independent arrays of samples recorded from ``start`` onward."""
        if not 0 <= start <= self.sample_count:
            raise ValueError("start must be within the recorded sample range")
        return {
            name: np.asarray(values[start:], dtype=float).copy()
            for name, values in self._samples.items()
        }


@runtime_checkable
class SnapshotSource(Protocol):
    """An opt-in value sampled at experiment-defined snapshot boundaries."""

    @property
    def report_name(self) -> str: ...

    def snapshot(self, snn: SNN) -> SignalValue: ...


@dataclass(frozen=True)
class ProjectionStrengthSnapshot:
    """Minimum, mean, and maximum compiled strength of one declared projection."""

    connection: ConnectionSpec

    @property
    def report_name(self) -> str:
        return f"strength:{self.connection.projection_name}"

    def snapshot(self, snn: SNN) -> np.ndarray:
        mask = snn.synapses.projection_mask(self.connection.projection_name)
        values = snn.synapses.strength[mask]
        if values.size == 0:
            raise ValueError(
                f"projection {self.connection.projection_name!r} has no compiled synapses"
            )
        return np.array([values.min(), values.mean(), values.max()], dtype=float)


__all__ = [
    "ProjectionStrengthSnapshot",
    "SignalRecorder",
    "SnapshotSource",
]
