"""Affective populations that live inside the main spiking network."""

from __future__ import annotations

import numpy as np

from continual_agent.cognition.affect import AffectiveState
from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses

AFFECT_SIGNALS = (
    "valence",
    "arousal",
    "uncertainty",
    "curiosity",
    "threat",
    "competence",
    "social_affiliation",
)


class AffectiveCircuit:
    """Population layout and local alignment for neural affect signals.

    The neurons themselves belong to ``SpikingNetwork``. This class only knows
    how affect populations are laid out, decoded, and aligned to teacher
    targets; it does not run a second neural simulator.
    """

    signal_names = AFFECT_SIGNALS

    def __init__(self, neurons_per_signal: int = 4, learning_rate: float = 0.12):
        if neurons_per_signal <= 0:
            raise ValueError("neurons_per_signal must be positive")
        self.neurons_per_signal = neurons_per_signal
        self.learning_rate = learning_rate

    @property
    def neuron_count(self) -> int:
        return len(self.signal_names) * self.neurons_per_signal

    def groups(self, layout: PopulationLayout) -> dict[str, np.ndarray]:
        bounds = layout.slice(Population.AFFECT)
        if bounds.stop - bounds.start != self.neuron_count:
            raise ValueError("layout affect population does not match circuit")
        return {
            name: np.arange(
                layout.subgroup(Population.AFFECT, name).start,
                layout.subgroup(Population.AFFECT, name).stop,
            )
            for name in self.signal_names
        }

    def projection_indices(self, layout: PopulationLayout) -> np.ndarray:
        """Return ``[signal, input_feature, neuron_in_population]`` edge IDs."""

        bounds = layout.slice(Population.AFFECT)
        input_count = layout.input_count
        if bounds.stop - bounds.start != self.neuron_count:
            raise ValueError("layout affect population does not match circuit")
        indices = np.empty(
            (len(self.signal_names), input_count, self.neurons_per_signal),
            dtype=np.int64,
        )
        affect_count = self.neuron_count
        for signal_index in range(len(self.signal_names)):
            for feature in range(input_count):
                edge_start = feature * affect_count + signal_index * self.neurons_per_signal
                indices[signal_index, feature] = np.arange(
                    edge_start, edge_start + self.neurons_per_signal
                )
        return indices

    def _targets(self, state: AffectiveState | dict[str, float]) -> np.ndarray:
        values = state.as_dict() if isinstance(state, AffectiveState) else state
        targets = np.asarray([values[name] for name in self.signal_names], dtype=float)
        targets[0] = (targets[0] + 1.0) / 2.0
        return np.clip(targets, 0.0, 1.0)

    def projection_prediction(
        self,
        synapses: SparseSynapses,
        edge_indices: np.ndarray,
        features: np.ndarray,
    ) -> dict[str, float]:
        features = np.asarray(features, dtype=float)
        weights = synapses.weight[edge_indices].mean(axis=2)
        probabilities = 1.0 / (1.0 + np.exp(-(weights @ features)))
        values = probabilities.copy()
        values[0] = 2.0 * values[0] - 1.0
        return {name: float(value) for name, value in zip(self.signal_names, values)}

    def align(
        self,
        synapses: SparseSynapses,
        edge_indices: np.ndarray,
        features: np.ndarray,
        state: AffectiveState | dict[str, float],
    ) -> None:
        """Apply local feature/target alignment to input-to-affect synapses."""

        features = np.asarray(features, dtype=float)
        weights = synapses.weight[edge_indices].mean(axis=2)
        probabilities = 1.0 / (1.0 + np.exp(-(weights @ features)))
        error = self._targets(state) - probabilities
        update = self.learning_rate * error[:, None] * features[None, :]
        synapses.weight[edge_indices] += update[:, :, None] / self.neurons_per_signal
        synapses.weight[edge_indices] = np.clip(synapses.weight[edge_indices], -1.0, 1.0)

    def decode(self, spike_frames: list[np.ndarray], layout: PopulationLayout) -> dict[str, float]:
        if not spike_frames:
            return {name: 0.0 for name in self.signal_names}
        groups = self.groups(layout)
        rates = np.asarray(
            [
                sum(frame[group].sum() for frame in spike_frames)
                / (len(spike_frames) * self.neurons_per_signal)
                for group in groups.values()
            ]
        )
        values = np.clip(rates, 0.0, 1.0)
        values[0] = 2.0 * values[0] - 1.0
        return {name: float(value) for name, value in zip(self.signal_names, values)}
