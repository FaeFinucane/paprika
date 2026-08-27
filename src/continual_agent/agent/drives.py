"""Composable external-current sources for a network tick."""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Entire drive protocol is too abstract and difficult to follow.
# Especially with ArrayDrive being the way the actual current is modified?!
# Remove as abstraction and just ensure background current and homeostasis are well encapsulated but
# called directly in SpikingRuntime

class Drive(Protocol):
    """Add this tick's current to ``output`` without allocating it."""

    def add_to(self, output: np.ndarray) -> None: ...


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
        self._initial_rng_state = self.rng.bit_generator.state
        self._sample = np.zeros(size, dtype=float)
        self._random = np.zeros(size, dtype=float)

    def add_to(self, output: np.ndarray) -> None:
        if self.rate:
            self.rng.random(out=self._random)
            np.less(self._random, self.rate, out=self._random)
            np.multiply(self._random, self.current, out=self._sample)
            np.add(output, self._sample, out=output)

    def state_snapshot(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "current": self.current,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, snapshot: dict[str, object]) -> None:
        rate = float(snapshot["rate"])  # type: ignore[arg-type]
        current = float(snapshot["current"])  # type: ignore[arg-type]
        if not np.isfinite(rate) or not 0 <= rate <= 1 or not np.isfinite(current) or current < 0:
            raise ValueError("invalid background drive state")
        self.rate = rate
        self.current = current
        self.rng.bit_generator.state = snapshot["rng_state"]  # type: ignore[assignment]

    def reset_rng(self) -> None:
        """Return stochastic sampling to the configured stream origin."""
        self.rng.bit_generator.state = self._initial_rng_state
