"""Windowed, task-neutral diagnostics for the spiking runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_agent.simulation.population_layout import Population, PopulationLayout


@dataclass(frozen=True)
class PopulationDiagnostics:
    """Activity statistics for one population over the current window.

    ``active_fraction`` is the fraction of neurons that emitted at least once;
    ``silent_fraction`` emitted zero times; ``saturated_fraction`` reached the
    configurable saturation rate. These fractions are mutually exhaustive.
    Rates are spikes per neuron per simulation tick.
    """

    neuron_count: float
    spike_count: float
    firing_rate_mean: float
    firing_rate_spread: float
    active_fraction: float
    silent_fraction: float
    saturated_fraction: float
    voltage_mean: float
    voltage_spread: float
    voltage_min: float
    voltage_max: float
    threshold_mean: float
    threshold_spread: float
    threshold_min: float
    threshold_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "neuron_count": self.neuron_count,
            "spike_count": self.spike_count,
            "firing_rate_mean": self.firing_rate_mean,
            "firing_rate_spread": self.firing_rate_spread,
            "active_fraction": self.active_fraction,
            "silent_fraction": self.silent_fraction,
            "saturated_fraction": self.saturated_fraction,
            "voltage_mean": self.voltage_mean,
            "voltage_spread": self.voltage_spread,
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
            "threshold_mean": self.threshold_mean,
            "threshold_spread": self.threshold_spread,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
        }


class RuntimeMetrics:
    """Accumulate spike and voltage observations without retaining all spikes."""

    def __init__(self, layout: PopulationLayout, saturation_rate: float = 0.5) -> None:
        if saturation_rate < 0.0:
            raise ValueError("saturation_rate must be non-negative")
        self.layout = layout
        self.saturation_rate = saturation_rate
        self.reset()

    def reset(self) -> None:
        self.ticks = 0
        self.output_events = 0
        self.spikes = np.zeros(self.layout.total_count)
        self.voltage_sum = np.zeros(self.layout.total_count)
        self.voltage_square_sum = np.zeros(self.layout.total_count)
        self.voltage_minimum = np.full(self.layout.total_count, np.inf)
        self.voltage_maximum = np.full(self.layout.total_count, -np.inf)

    def record(self, spikes: np.ndarray, voltage: np.ndarray) -> None:
        spikes = np.asarray(spikes, dtype=float)
        voltage = np.asarray(voltage, dtype=float)
        if spikes.shape != (self.layout.total_count,) or voltage.shape != spikes.shape:
            raise ValueError("spikes and voltage must match the runtime population size")
        self.ticks += 1
        self.spikes += spikes
        self.voltage_sum += voltage
        self.voltage_square_sum += voltage * voltage
        self.voltage_minimum = np.minimum(self.voltage_minimum, voltage)
        self.voltage_maximum = np.maximum(self.voltage_maximum, voltage)

    def record_output_event(self) -> None:
        self.output_events += 1

    @property
    def output_event_rate(self) -> float:
        return float(self.output_events / max(self.ticks, 1))

    def population(
        self, population: Population, threshold: float | np.ndarray
    ) -> PopulationDiagnostics:
        bounds = self.layout.slice(population)
        indices = np.arange(bounds.start, bounds.stop)
        ticks = max(self.ticks, 1)
        rates = self.spikes[indices] / ticks
        voltage_mean = self.voltage_sum[indices] / ticks
        voltage_variance = np.maximum(
            self.voltage_square_sum[indices] / ticks - voltage_mean * voltage_mean, 0.0
        )
        thresholds = np.broadcast_to(
            np.asarray(threshold, dtype=float), (self.layout.total_count,)
        )[indices]
        observed = self.ticks > 0
        return PopulationDiagnostics(
            float(rates.size),
            float(self.spikes[indices].sum()),
            float(rates.mean()) if rates.size else 0.0,
            float(rates.std()) if rates.size else 0.0,
            float(np.mean(rates > 0.0)) if rates.size else 0.0,
            float(np.mean(rates == 0.0)) if rates.size else 0.0,
            float(np.mean(rates >= self.saturation_rate)) if rates.size and observed else 0.0,
            float(voltage_mean.mean()) if rates.size else 0.0,
            float(np.sqrt(voltage_variance.mean())) if rates.size else 0.0,
            float(self.voltage_minimum[indices].min()) if rates.size and observed else 0.0,
            float(self.voltage_maximum[indices].max()) if rates.size and observed else 0.0,
            float(thresholds.mean()) if rates.size else 0.0,
            float(thresholds.std()) if rates.size else 0.0,
            float(thresholds.min()) if rates.size else 0.0,
            float(thresholds.max()) if rates.size else 0.0,
        )
