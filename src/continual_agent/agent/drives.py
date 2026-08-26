"""Composable external-current sources for a network tick."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Drive(Protocol):
    """Add this tick's current to ``output`` without allocating it."""

    def add_to(self, output: np.ndarray) -> None: ...


class DriveAggregator:
    """Combine registered drives into one preallocated current buffer."""

    def __init__(self, neuron_count: int) -> None:
        self.buffer = np.zeros(neuron_count, dtype=float)
        self.drives: list[Drive] = []

    def add(self, drive: Drive) -> None:
        self.drives.append(drive)

    def collect(self) -> np.ndarray:
        self.buffer.fill(0.0)
        for drive in self.drives:
            drive.add_to(self.buffer)
        return self.buffer


class ArrayDrive:
    """A mutable array drive, useful for task adapters and tests."""

    def __init__(self, size: int) -> None:
        self.current = np.zeros(size, dtype=float)

    def add_to(self, output: np.ndarray) -> None:
        np.add(output, self.current, out=output)


class BackgroundDrive:
    """Seeded stochastic current source and owner of its configuration."""

    def __init__(self, size: int, rate: float, current: float, seed: int) -> None:
        if not 0.0 <= rate <= 1.0 or current < 0.0:
            raise ValueError("background rate must be in [0, 1] and current must be non-negative")
        self.rate = rate
        self.current = current
        self.rng = np.random.default_rng(seed)
        self._sample = np.zeros(size, dtype=float)
        self._random = np.zeros(size, dtype=float)

    def add_to(self, output: np.ndarray) -> None:
        if self.rate:
            self.rng.random(out=self._random)
            np.less(self._random, self.rate, out=self._random)
            np.multiply(self._random, self.current, out=self._sample)
            np.add(output, self._sample, out=output)


class HomeostasisDrive:
    def __init__(self, homeostasis: object) -> None:
        self.homeostasis = homeostasis

    def add_to(self, output: np.ndarray) -> None:
        self.homeostasis.current_into(output)  # type: ignore[attr-defined]
