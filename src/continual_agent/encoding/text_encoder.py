"""Small deterministic text-to-current encoder.

This is intentionally not a language model. Hashed character n-grams give the
first experiments a repeatable, vocabulary-free representation while keeping
the neural network responsible for temporal association.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from continual_agent.agent.session import (
    BOUNDARY_CHANNEL_COUNT,
    InputSignal,
)


@dataclass(frozen=True)
class InputEvent:
    """One control or semantic input event in a presentation."""

    signal: InputSignal | None = None
    frame: np.ndarray | None = None


@dataclass(frozen=True)
class InputPresentationConfig:
    """Rate-coded presentation speed, measured in ticks per encoded frame."""

    presentation_speed: int = 1

    def __post_init__(self) -> None:
        if self.presentation_speed <= 0:
            raise ValueError("presentation_speed must be positive")


class TextEncoder:
    def __init__(
        self,
        feature_count: int,
        ticks_per_token: int = 3,
        current: float = 3.0,
        presentation_speed: int = 1,
    ):
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if feature_count <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("feature_count must leave room for boundary channels")
        if ticks_per_token <= 0:
            raise ValueError("ticks_per_token must be positive")
        self.feature_count = feature_count
        self.ticks_per_token = ticks_per_token
        self.current = current
        self.presentation = InputPresentationConfig(presentation_speed)

    def _feature(self, text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return BOUNDARY_CHANNEL_COUNT + (
            int.from_bytes(digest, "little") % (self.feature_count - BOUNDARY_CHANNEL_COUNT)
        )

    def encode(self, text: str, turn_marker: str = "USER_TURN") -> np.ndarray:
        """Return ``[ticks, feature_count]`` current frames."""

        normalised = " ".join(text.lower().split())
        symbols = [turn_marker.lower(), *normalised]
        if not normalised:
            symbols.append("<empty>")
        else:
            # Preserve a high-level identity feature as well as local n-grams.
            # This keeps tiny initial experiments separable without requiring
            # a learned tokenizer.
            symbols.append(f"<message:{normalised}>")

        frames: list[np.ndarray] = []
        for index, symbol in enumerate(symbols):
            features = {self._feature(symbol)}
            if index:
                features.add(self._feature(symbols[index - 1] + symbol))
            frame = np.zeros(self.feature_count, dtype=float)
            frame[list(features)] = self.current
            frames.extend([frame] * self.ticks_per_token)
        return np.asarray(frames)

    def feature_vector(self, text: str) -> np.ndarray:
        """Return a stable sparse identity code for the input message."""

        normalised = " ".join(text.lower().split()) or "<empty>"
        vector = np.zeros(self.feature_count, dtype=float)
        vector[self._feature(f"<message:{normalised}>")] = 1.0
        vector[self._feature("<turn:user>")] = 0.1
        return vector

    def present(
        self,
        text: str,
        *,
        presentation_speed: int | None = None,
    ) -> tuple[InputEvent, ...]:
        """Present semantic frames with explicit input boundaries.

        ``encode`` remains the semantic encoder. This layer only changes its
        temporal presentation: rate coding repeats activity for the configured
        number of ticks.
        """

        config = InputPresentationConfig(
            presentation_speed
            if presentation_speed is not None
            else self.presentation.presentation_speed
        )
        semantic_frames = self.encode(text)
        # ``encode`` supplies base semantic timing; presentation speed stretches it.
        ticks = config.presentation_speed
        events: list[InputEvent] = [InputEvent(signal=InputSignal.INPUT_BEGIN)]
        for frame in semantic_frames:
            events.extend(InputEvent(frame=np.array(frame, copy=True)) for _ in range(ticks))
        events.append(InputEvent(signal=InputSignal.INPUT_END))
        return tuple(events)
