"""Slow bounded population-level homeostatic current regulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_agent.simulation.population_layout import Population, PopulationLayout


@dataclass(frozen=True)
class HomeostasisConfig:
    # Enabled should not be part of config. If homeostasis is disabled, HomeoStasisConfig should be None, and PopulationHomeostasis should not be created.
    enabled: bool = False
    target_rate: float = 0.1
    strength: float = 0.01
    update_interval: int = 100
    max_current: float = 0.25
    populations: tuple[Population, ...] = (Population.HIDDEN,)


class PopulationHomeostasis:
    """Adjust one shared current per population, never individual neurons."""

    def __init__(
        self,
        layout: PopulationLayout,
        config: HomeostasisConfig | None = None,
    ) -> None:
        config = config or HomeostasisConfig()
        if (
            not 0.0 <= config.target_rate
            or config.strength < 0.0
            or config.update_interval <= 0
            or config.max_current < 0.0
        ):
            raise ValueError("invalid homeostasis configuration")
        self.layout = layout
        self.config = config
        self.drive = np.zeros(layout.total_count)
        self._ticks = 0
        self._spikes = np.zeros(layout.total_count)

    def reset(self) -> None:
        self.drive.fill(0.0)
        self._ticks = 0
        self._spikes.fill(0.0)

    def add_to(self, output: np.ndarray) -> None:
        np.add(output, self.drive, out=output)

    def observe(self, spikes: np.ndarray) -> None:
        if not self.config.enabled:
            return
        # Why not just use the tick passed in by NetworkPlugin, and do if tick modulo update_interval == 0 and tick > 0?
        self._ticks += 1
        self._spikes += np.asarray(spikes, dtype=float)
        if self._ticks < self.config.update_interval:
            return
        for population in self.config.populations:
            bounds = self.layout.slice(population)
            rate = self._spikes[bounds].mean() / self._ticks
            adjustment = self.config.strength * (self.config.target_rate - rate)
            self.drive[bounds] = np.clip(
                self.drive[bounds] + adjustment, -self.config.max_current, self.config.max_current
            )
        self._ticks = 0
        self._spikes.fill(0.0)
