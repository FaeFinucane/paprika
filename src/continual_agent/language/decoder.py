"""A tiny online character decoder with local associative learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodedText:
    text: str
    tokens: tuple[str, ...]
    stopped_on_eos: bool


class CharacterDecoder:
    """Character/byte-like decoder using a local context-to-token table.

    This is intentionally small and transparent. It is not a transformer and
    has no offline backpropagation phase. Context statistics are updated after
    each observed next character, so replay and live learning use the same
    mechanism.
    """

    DEFAULT_ALPHABET = tuple(" abcdefghijklmnopqrstuvwxyz.,!?'-\n")

    def __init__(
        self,
        context_size: int = 3,
        alphabet: tuple[str, ...] = DEFAULT_ALPHABET,
        learning_rate: float = 0.15,
        seed: int = 0,
    ):
        if context_size <= 0 or not alphabet:
            raise ValueError("context_size and alphabet must be positive")
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must not contain duplicates")
        self.context_size = context_size
        self.alphabet = alphabet
        self.eos = "<EOS>"
        self.tokens = alphabet + (self.eos,)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self._token_index = {token: index for index, token in enumerate(self.tokens)}
        self._counts = np.zeros(
            (len(alphabet) ** context_size, len(self.tokens)), dtype=float
        )

    def _context_index(self, context: str) -> int:
        context = context[-self.context_size:].rjust(self.context_size)
        index = 0
        for character in context:
            index *= len(self.alphabet)
            index += self._token_index.get(character, 0)
        return index

    def observe(self, text: str, include_eos: bool = True) -> None:
        """Learn observed next-character transitions online."""

        clean = "".join(character for character in text.lower() if character in self._token_index)
        context = "".rjust(self.context_size)
        sequence = list(clean) + ([self.eos] if include_eos else [])
        for token in sequence:
            row = self._context_index(context)
            self._counts[row, self._token_index[token]] += self.learning_rate
            context = (context + token)[-self.context_size:]

    def _distribution(self, context: str) -> np.ndarray:
        row = self._counts[self._context_index(context)].copy()
        # A small non-zero prior avoids dead contexts while preserving learned
        # associations and deterministic seeded exploration.
        row += 0.01
        row[-1] *= 0.5
        return row / row.sum()

    def next_token(self, context: str, sample: bool = False) -> str:
        distribution = self._distribution(context)
        if sample:
            return self.tokens[int(self.rng.choice(len(self.tokens), p=distribution))]
        return self.tokens[int(np.argmax(distribution))]

    def generate(self, prompt: str = "", max_tokens: int = 80, sample: bool = False) -> DecodedText:
        context = "".join(character for character in prompt.lower() if character in self._token_index)
        emitted: list[str] = []
        stopped_on_eos = False
        for _ in range(max_tokens):
            token = self.next_token(context, sample=sample)
            if token == self.eos:
                stopped_on_eos = True
                break
            emitted.append(token)
            context = (context + token)[-self.context_size:]
        return DecodedText("".join(emitted), tuple(emitted), stopped_on_eos)

    def loss(self, text: str) -> float:
        """Return mean negative log likelihood for simple held-out evaluation."""

        clean = "".join(character for character in text.lower() if character in self._token_index)
        context = "".rjust(self.context_size)
        losses: list[float] = []
        for token in list(clean) + [self.eos]:
            probability = self._distribution(context)[self._token_index[token]]
            losses.append(-float(np.log(max(probability, 1e-12))))
            context = (context + token)[-self.context_size:]
        return float(np.mean(losses)) if losses else 0.0
