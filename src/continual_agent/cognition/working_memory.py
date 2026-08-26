"""Small persistent recurrent working-memory state for multi-turn dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkingMemory:
    """A bounded, decaying vector used across turns in one conversation."""

    size: int
    decay: float = 0.96
    state: np.ndarray = field(init=False)
    turns: int = 0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.state = np.zeros(self.size, dtype=float)

    def update(self, features: np.ndarray, salience: float = 1.0) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.shape != (self.size,):
            raise ValueError(f"features must have shape ({self.size},)")
        self.state = self.decay * self.state + float(salience) * features
        self.state = np.clip(self.state, -1.0, 1.0)
        self.turns += 1
        return self.state.copy()

    def context_current(self, gain: float = 0.8) -> np.ndarray:
        return gain * self.state

    def reset(self) -> None:
        self.state.fill(0.0)
        self.turns = 0

    def snapshot(self) -> dict[str, object]:
        return {"turns": self.turns, "state": self.state.tolist()}
