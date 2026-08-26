"""Small deterministic text-to-current encoder.

This is intentionally not a language model. Hashed character n-grams give the
first experiments a repeatable, vocabulary-free representation while keeping
the neural network responsible for temporal association.
"""

from __future__ import annotations

import hashlib

import numpy as np


class TextEncoder:
    def __init__(self, feature_count: int, ticks_per_token: int = 3, current: float = 3.0):
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if ticks_per_token <= 0:
            raise ValueError("ticks_per_token must be positive")
        self.feature_count = feature_count
        self.ticks_per_token = ticks_per_token
        self.current = current

    def _feature(self, text: str, ngram: int) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.feature_count

    def encode(self, text: str, turn_marker: str = "USER_TURN") -> np.ndarray:
        """Return ``[ticks, feature_count]`` current frames."""

        normalised = " ".join(text.lower().split())
        symbols = [turn_marker.lower(), *normalised]
        if not normalised:
            symbols.append("<empty>")
        else:
            # Preserve a high-level identity feature as well as local n-grams.
            # This keeps the tiny first curriculum separable without requiring
            # a learned tokenizer.
            symbols.append(f"<message:{normalised}>")

        frames: list[np.ndarray] = []
        for index, symbol in enumerate(symbols):
            features = {self._feature(symbol, 1)}
            if index:
                features.add(self._feature(symbols[index - 1] + symbol, 2))
            frame = np.zeros(self.feature_count, dtype=float)
            frame[list(features)] = self.current
            frames.extend([frame.copy() for _ in range(self.ticks_per_token)])
        return np.asarray(frames)

    def feature_vector(self, text: str) -> np.ndarray:
        """Return a stable sparse identity code for local policy learning."""

        normalised = " ".join(text.lower().split()) or "<empty>"
        vector = np.zeros(self.feature_count, dtype=float)
        vector[self._feature(f"<message:{normalised}>", 1)] = 1.0
        vector[self._feature("<turn:user>", 1)] = 0.1
        return vector
