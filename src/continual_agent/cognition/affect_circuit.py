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

    The neurons themselves belong to ``NetworkCore``. This class only knows
    how affect populations are laid out, decoded, and aligned to teacher
    targets; it does not run a second neural simulator.
    """

    signal_names = AFFECT_SIGNALS

    def __init__(self, layout: PopulationLayout, learning_rate: float = 0.12):
        if tuple(layout.affect_subgroups) != self.signal_names:
            raise ValueError("layout affect subgroups must match affect signals")
        self.layout = layout
        if not np.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        self.learning_rate = learning_rate

    def groups(self, layout: PopulationLayout) -> dict[str, np.ndarray]:
        return {
            name: np.arange(
                layout.subgroup(Population.AFFECT, name).start,
                layout.subgroup(Population.AFFECT, name).stop,
            )
            for name in self.signal_names
        }

    def projection_indices(self, layout: PopulationLayout) -> np.ndarray:
        """Return ``[signal, input_feature, neuron_in_population]`` edge IDs."""

        input_count = layout.input_count
        indices = np.empty(
            (len(self.signal_names), input_count, layout.subgroup_width(Population.AFFECT)),
            dtype=np.int64,
        )
        affect_count = layout.affect_count
        for signal_index in range(len(self.signal_names)):
            for feature in range(input_count):
                edge_start = feature * affect_count + signal_index * layout.subgroup_width(
                    Population.AFFECT
                )
                indices[signal_index, feature] = np.arange(
                    edge_start, edge_start + layout.subgroup_width(Population.AFFECT)
                )
        return indices

    def _targets(self, state: AffectiveState | dict[str, float]) -> np.ndarray:
        values = state.as_dict() if isinstance(state, AffectiveState) else state
        if any(name not in values for name in self.signal_names):
            raise ValueError("affect targets are missing a circuit signal")
        targets = np.asarray([values[name] for name in self.signal_names], dtype=float)
        if targets.shape != (len(self.signal_names),) or not np.isfinite(targets).all():
            raise ValueError("affect targets must be finite scalars")
        targets[0] = (targets[0] + 1.0) / 2.0
        return np.clip(targets, 0.0, 1.0)

    def projection_prediction(
        self,
        synapses: SparseSynapses,
        edge_indices: np.ndarray,
        features: np.ndarray,
    ) -> dict[str, float]:
        features = np.asarray(features, dtype=float)
        self._validate_projection_inputs(synapses, edge_indices, features)
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
        self._validate_projection_inputs(synapses, edge_indices, features)
        weights = synapses.weight[edge_indices].mean(axis=2)
        probabilities = 1.0 / (1.0 + np.exp(-(weights @ features)))
        error = self._targets(state) - probabilities
        update = self.learning_rate * error[:, None] * features[None, :]
        synapses.weight[edge_indices] += update[:, :, None] / edge_indices.shape[2]
        synapses.weight[edge_indices] = np.clip(synapses.weight[edge_indices], -1.0, 1.0)

    def _validate_projection_inputs(
        self, synapses: SparseSynapses, edge_indices: np.ndarray, features: np.ndarray
    ) -> None:
        indices = np.asarray(edge_indices)
        if indices.dtype.kind not in "iu":
            raise ValueError("affect projection edge indices must be integers")
        if indices.shape != (
            len(self.signal_names),
            features.size,
            self.layout.subgroup_width(Population.AFFECT),
        ):
            raise ValueError("affect projection indices have the wrong shape")
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("affect features must be a finite 1D array")
        if np.any(indices < 0) or np.any(indices >= synapses.weight.size):
            raise ValueError("affect projection edge index is out of bounds")

    def decode(self, spike_frames: list[np.ndarray], layout: PopulationLayout) -> dict[str, float]:
        if not spike_frames:
            return {name: 0.0 for name in self.signal_names}
        for frame in spike_frames:
            frame = np.asarray(frame)
            if frame.shape != (layout.total_count,):
                raise ValueError("affect spike frames must match the layout size")
            if not np.isfinite(frame).all():
                raise ValueError("affect spike frames must be finite")
        groups = self.groups(layout)
        rates = np.asarray(
            [
                sum(frame[group].sum() for frame in spike_frames)
                / (len(spike_frames) * layout.subgroup_width(Population.AFFECT))
                for group in groups.values()
            ]
        )
        values = np.clip(rates, 0.0, 1.0)
        values[0] = 2.0 * values[0] - 1.0
        return {name: float(value) for name, value in zip(self.signal_names, values)}
