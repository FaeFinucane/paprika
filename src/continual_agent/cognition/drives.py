"""Slow internal variables that modulate behaviour and plasticity."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriveState:
    """Homeostatic drive values in the normalised range [0, 1]."""

    values: dict[str, float] = field(
        default_factory=lambda: {
            "curiosity": 0.5,
            "uncertainty": 0.5,
            "safety": 0.0,
            "coherence": 0.5,
            "competence": 0.5,
        }
    )
    learning_rate: float = 0.1

    def update(self, changes: dict[str, float]) -> None:
        for name, change in changes.items():
            current = self.values.get(name, 0.5)
            self.values[name] = min(1.0, max(0.0, current + self.learning_rate * change))

    def modulation(self) -> float:
        """Return a bounded global learning/attention multiplier."""

        uncertainty = self.values.get("uncertainty", 0.5)
        competence = self.values.get("competence", 0.5)
        return 0.5 + 0.75 * uncertainty + 0.25 * competence
