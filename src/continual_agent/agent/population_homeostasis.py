"""Slow bounded population-level homeostatic current regulation."""

from __future__ import annotations

import numpy as np

from continual_agent.simulation.population_layout import Population, PopulationLayout


class PopulationHomeostasis:
    """Adjust one shared current per population, never individual neurons."""

    def __init__(
        self,
        layout: PopulationLayout,
        *,
        enabled: bool = False,
        target_rate: float = 0.1,
        strength: float = 0.01,
        update_interval: int = 100,
        max_current: float = 0.25,
        populations: tuple[Population, ...] = (Population.HIDDEN,),
    ) -> None:
        if not 0.0 <= target_rate or strength < 0.0 or update_interval <= 0 or max_current < 0.0:
            raise ValueError("invalid homeostasis configuration")
        self.layout = layout
        self.enabled = enabled
        self.target_rate = target_rate
        self.strength = strength
        self.update_interval = update_interval
        self.max_current = max_current
        self.populations = populations
        self.drive = np.zeros(layout.total_count)
        self._ticks = 0
        self._spikes = np.zeros(layout.total_count)

    def reset(self) -> None:
        self.drive.fill(0.0)
        self._ticks = 0
        self._spikes.fill(0.0)

    def current(self) -> np.ndarray:
        return self.drive.copy()

    def current_into(self, output: np.ndarray) -> None:
        np.add(output, self.drive, out=output)

    def observe(self, spikes: np.ndarray) -> None:
        if not self.enabled:
            return
        self._ticks += 1
        self._spikes += np.asarray(spikes, dtype=float)
        if self._ticks < self.update_interval:
            return
        for population in self.populations:
            bounds = self.layout.slice(population)
            rate = self._spikes[bounds].mean() / self._ticks
            adjustment = self.strength * (self.target_rate - rate)
            self.drive[bounds] = np.clip(
                self.drive[bounds] + adjustment, -self.max_current, self.max_current
            )
        self._ticks = 0
        self._spikes.fill(0.0)
