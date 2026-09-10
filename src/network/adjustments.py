"""Proposed, transactional changes to long-lived network parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkAdjustment:
    """One rule's proposed changes, committed with other rules as a batch."""

    strength_delta: np.ndarray | None = None
    strength_mask: np.ndarray | None = None
    def __post_init__(self) -> None:
        if (self.strength_delta is None) != (self.strength_mask is None):
            raise ValueError("strength delta and mask must be supplied together")
