"""Spiking character output populations and local sequence plasticity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        alphabet: tuple[str, ...] = DEFAULT_ALPHABET,
        neurons_per_token: int = 3,
        context_size: int = 3,
        learning_rate: float = 0.08,
    ):
        if not alphabet or neurons_per_token <= 0 or context_size <= 0:
            raise ValueError("alphabet, neurons_per_token, and context_size are required")
        self.alphabet = alphabet
        self.tokens = ("<EOS>",) + alphabet
        self.neurons_per_token = neurons_per_token
        self.context_size = context_size
        self.learning_rate = learning_rate

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def neuron_count(self) -> int:
        return self.token_count * self.neurons_per_token

    def groups(self, start: int) -> dict[str, np.ndarray]:
        return {
            token: np.arange(
                start + index * self.neurons_per_token,
                start + (index + 1) * self.neurons_per_token,
            )
            for index, token in enumerate(self.tokens)
        }

    def output_score(
        self, spike_frames: list[np.ndarray], start: int
    ) -> dict[str, float]:
        groups = self.groups(start)
        denominator = max(1, len(spike_frames) * self.neurons_per_token)
        return {
            token: float(sum(frame[group].sum() for frame in spike_frames) / denominator)
            for token, group in groups.items()
        }

    def choose(self, spike_frames: list[np.ndarray], start: int) -> str:
        scores = self.output_score(spike_frames, start)
        return max(scores, key=lambda token: scores[token])

    def choose_non_eos(
        self, spike_frames: list[np.ndarray], start: int
    ) -> str:
        scores = self.output_score(spike_frames, start)
        scores.pop("<EOS>")
        return max(scores, key=lambda token: scores[token])

    def token_projection_indices(
        self,
        input_count: int,
        source_start: int,
        token_start: int,
    ) -> np.ndarray:
        """Return edge IDs for input/decoder-state to token projections.

        The returned IDs are relative to the projection's own appended edge
        block; the agent offsets them when adding the block to the network.
        """

        indices = np.empty(
            (input_count, self.token_count, self.neurons_per_token),
            dtype=np.int64,
        )
        for feature in range(input_count):
            for token in range(self.token_count):
                start = feature * self.neuron_count + token * self.neurons_per_token
                indices[feature, token] = np.arange(
                    start, start + self.neurons_per_token
                )
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
        synapses.weight[selected_edges] += update[:, None] / self.neurons_per_token
        synapses.weight[selected_edges] = np.clip(
            synapses.weight[selected_edges], -1.0, 1.0
        )

    def target_tokens(self, text: str) -> tuple[str, ...]:
        clean = "".join(character for character in text.lower() if character in self.alphabet)
        return tuple(clean) + ("<EOS>",)
