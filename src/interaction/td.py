"""Pure, deterministic temporal-difference reference calculations.

This module deliberately has no dependency on channels, spikes, or the network.  It
is useful as an oracle when checking an implementation of a positive/negative TD
signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Hashable


def _number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _magnitude(name: str, value: float) -> float:
    result = _number(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative for unipolar coding")
    return result


@dataclass(frozen=True, slots=True)
class TDComparison:
    """The two independently calculated TD terms and their signed comparison."""

    positive_term: float
    negative_term: float
    delta: float

    @property
    def td_positive(self) -> float:
        return self.positive_term

    @property
    def td_negative(self) -> float:
        return self.negative_term


@dataclass(slots=True)
class TemporalDifferenceComparator:
    """Reference two-channel TD comparator.

    For each transition, terms are calculated as ``r + gamma * V_now - V_prev``.
    ``delta`` is the positive term minus the negative term.  Stored previous
    values are updated only after both terms have been calculated, exactly once.

    ``transition_id`` is optional, but when supplied it provides a guard against
    accidentally evaluating the same transition twice.  IDs are only meaningful
    until :meth:`reset` is called.
    """

    previous_positive: float = 0.0
    previous_negative: float = 0.0
    _seen_transition_ids: set[Hashable] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.previous_positive = _magnitude("previous_positive", self.previous_positive)
        self.previous_negative = _magnitude("previous_negative", self.previous_negative)

    def evaluate_transition(
        self,
        r_positive: float,
        r_negative: float,
        v_positive_now: float,
        v_negative_now: float,
        gamma: float,
        *,
        transition_id: Hashable | None = None,
    ) -> TDComparison:
        """Evaluate one transition and advance both previous values once."""
        rp = _magnitude("r_positive", r_positive)
        rn = _magnitude("r_negative", r_negative)
        vp = _magnitude("v_positive_now", v_positive_now)
        vn = _magnitude("v_negative_now", v_negative_now)
        discount = _number("gamma", gamma)
        if not 0.0 <= discount <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive")
        if transition_id is not None:
            try:
                if self._seen_transition_ids is None:
                    self._seen_transition_ids = set()
                if transition_id in self._seen_transition_ids:
                    raise ValueError(f"transition_id {transition_id!r} was already evaluated")
                self._seen_transition_ids.add(transition_id)
            except TypeError as exc:
                raise TypeError("transition_id must be hashable") from exc

        positive = rp + discount * vp - self.previous_positive
        negative = rn + discount * vn - self.previous_negative
        result = TDComparison(positive, negative, positive - negative)
        self.previous_positive = vp
        self.previous_negative = vn
        return result

    def reset(self, previous_positive: float = 0.0, previous_negative: float = 0.0) -> None:
        """Reset the comparator state (and transition-ID guard) to a baseline."""
        self.previous_positive = _magnitude("previous_positive", previous_positive)
        self.previous_negative = _magnitude("previous_negative", previous_negative)
        self._seen_transition_ids = None
