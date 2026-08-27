"""Spiking character output populations and local sequence plasticity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    tokens: tuple[str, ...]
    stopped_on_eos: bool
    ticks: int


class SpikingCharacterDecoder:
    """Layout and learning utilities for character populations in one network."""

    DEFAULT_ALPHABET = tuple(" abcdefghijklmnopqrstuvwxyz.,!?'-\n")

    def __init__(
        self,
        layout: PopulationLayout,
        learning_rate: float = 0.08,
    ):
        self.layout = layout
        self.tokens = layout.subgroup_names(Population.OUTPUT_CHAR)
        self.alphabet = tuple(token for token in self.tokens if token != "<EOS>")
        self.learning_rate = learning_rate

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def groups(self, layout: PopulationLayout) -> dict[str, np.ndarray]:
        return {
            token: np.arange(
                layout.subgroup(Population.OUTPUT_CHAR, token).start,
                layout.subgroup(Population.OUTPUT_CHAR, token).stop,
            )
            for token in self.tokens
        }

    def token_projection_indices(
        self,
        layout: PopulationLayout,
    ) -> np.ndarray:
        """Return edge IDs for input/decoder-state to token projections.

        The returned IDs are relative to the projection's own appended edge
        block; the agent offsets them when adding the block to the network.
        """

        if tuple(layout.char_subgroups) != self.tokens:
            raise ValueError("layout output_char subgroups must match decoder tokens")
        token_bounds = layout.slice(Population.OUTPUT_CHAR)
        input_count = token_bounds.start
        # Edge IDs are relative to the appended block, but these bounds ensure
        # the IDs describe the supplied source and target populations.
        indices = np.empty(
            (input_count, self.token_count, layout.subgroup_width(Population.OUTPUT_CHAR)),
            dtype=np.int64,
        )
        for feature in range(input_count):
            for token in range(self.token_count):
                subgroup = layout.subgroup(Population.OUTPUT_CHAR, self.tokens[token])
                start = feature * layout.char_count + token * layout.subgroup_width(
                    Population.OUTPUT_CHAR
                )
                indices[feature, token] = np.arange(start, start + subgroup.stop - subgroup.start)
        return indices

    def align_next_token(
        self,
        synapses: SparseSynapses,
        edge_indices: np.ndarray,
        active_features: np.ndarray,
        target: str,
    ) -> None:
        """Locally strengthen the target token for active presynaptic features."""

        active_features = np.asarray(active_features, dtype=float)
        if active_features.shape != (edge_indices.shape[0],):
            raise ValueError("active_features has the wrong shape")
        target_index = self.tokens.index(target)
        update = self.learning_rate * active_features
        selected_edges = edge_indices[:, target_index]
        synapses.weight[selected_edges] += update[:, None] / edge_indices.shape[2]
        synapses.weight[selected_edges] = np.clip(synapses.weight[selected_edges], -1.0, 1.0)

    def align_hidden_output_token(
        self,
        synapses: SparseSynapses,
        edge_indices: np.ndarray,
        source: np.ndarray,
        target: str,
        layout: PopulationLayout,
    ) -> None:
        """Strengthen existing hidden-to-output edges into the teacher event.

        ``source`` is the activity observed in the network immediately before
        the teacher event.  The edge list is supplied by the network owner so
        this method cannot create a hidden language state or a new feedback
        projection.
        """
        edge_indices = np.asarray(edge_indices, dtype=np.int64)
        source = np.asarray(source, dtype=float)
        if edge_indices.ndim != 1:
            raise ValueError("edge_indices must be 1D")
        if source.shape != (synapses.neuron_count,):
            raise ValueError("source has the wrong shape")
        if not np.isfinite(source).all():
            raise ValueError("source must be finite")
        target_start = layout.subgroup(Population.OUTPUT_CHAR, target).start
        targets = synapses.target[edge_indices]
        target_bounds = np.arange(
            target_start, layout.subgroup(Population.OUTPUT_CHAR, target).stop
        )
        selected = edge_indices[np.isin(targets, target_bounds)]
        if selected.size == 0:
            return
        update = self.learning_rate * np.maximum(source[synapses.source[selected]], 0.0)
        synapses.weight[selected] = np.clip(synapses.weight[selected] + update, -1.0, 1.0)

    def target_tokens(self, text: str) -> tuple[str, ...]:
        clean = "".join(character for character in text.lower() if character in self.alphabet)
        return tuple(clean) + ("<EOS>",)
