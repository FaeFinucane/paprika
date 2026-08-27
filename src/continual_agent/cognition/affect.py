"""Bounded, inspectable affective and motivational state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class AffectiveEvent:
    """An externally observable event that may alter affective state."""

    reward_prediction_error: float = 0.0
    novelty: float = 0.0
    learning_progress: float = 0.0
    threat: float = 0.0
    urgency: float = 0.0
    uncertainty: float = 0.0
    correction: bool = False
    social_feedback: float = 0.0


@dataclass
class AffectiveState:
    """Functional affective state, not a claim about subjective experience."""

    valence: float = 0.0
    arousal: float = 0.2
    uncertainty: float = 0.5
    curiosity: float = 0.5
    threat: float = 0.1
    competence: float = 0.5
    social_affiliation: float = 0.5
    energy: float = 1.0
    update_rate: float = 0.1
    decay_rate: float = 0.03
    _baselines: dict[str, float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        values = [getattr(self, item.name) for item in fields(self) if item.name != "_baselines"]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("affective state values must be finite")
        if self.update_rate < 0 or self.decay_rate < 0:
            raise ValueError("affect rates must be non-negative")
        baseline_names = {
            "valence",
            "arousal",
            "uncertainty",
            "curiosity",
            "threat",
            "competence",
            "social_affiliation",
            "energy",
        }
        if self._baselines is None:
            self._baselines = {name: float(getattr(self, name)) for name in baseline_names}
        elif set(self._baselines) != baseline_names or not all(
            math.isfinite(value) for value in self._baselines.values()
        ):
            raise ValueError("affective baselines must contain finite state fields")
        assert self._baselines is not None
        self._bound_state()

    def _bound_state(self) -> None:
        self.valence = _clip(self.valence, -1.0, 1.0)
        for name in (
            "arousal",
            "uncertainty",
            "curiosity",
            "threat",
            "competence",
            "social_affiliation",
            "energy",
        ):
            setattr(self, name, _clip(getattr(self, name)))

    def advance(self, steps: int = 1) -> None:
        """Decay transient state and consume a small amount of energy."""

        if steps < 0:
            raise ValueError("steps cannot be negative")
        baselines = self._baselines
        assert baselines is not None
        for _ in range(steps):
            for name, baseline in baselines.items():
                if name == "energy":
                    continue
                value = getattr(self, name)
                setattr(self, name, value + self.decay_rate * (baseline - value))
            self.energy = _clip(self.energy - 0.002)
        self._bound_state()

    def reset(self) -> None:
        """Return affect to its configured baseline."""

        baselines = self._baselines
        assert baselines is not None
        for name, baseline in baselines.items():
            setattr(self, name, baseline)
        self._bound_state()

    def observe(self, event: AffectiveEvent) -> None:
        """Update state from an event and its delayed outcome."""

        event_values = (
            event.reward_prediction_error,
            event.novelty,
            event.learning_progress,
            event.threat,
            event.urgency,
            event.uncertainty,
            event.social_feedback,
        )
        if not all(math.isfinite(value) for value in event_values):
            raise ValueError("affective event values must be finite")
        rpe = max(-1.0, min(1.0, event.reward_prediction_error))
        threat = _clip(event.threat)
        urgency = _clip(event.urgency)
        novelty = _clip(event.novelty)
        progress = _clip(event.learning_progress)
        uncertainty = _clip(event.uncertainty)

        self.valence += self.update_rate * rpe
        self.competence += self.update_rate * (rpe if rpe >= 0 else 0.5 * rpe)
        self.threat += self.update_rate * (threat - self.threat)
        self.arousal += self.update_rate * (max(urgency, threat) - self.arousal)
        self.uncertainty += self.update_rate * (uncertainty - self.uncertainty)
        self.curiosity += self.update_rate * (novelty * progress - self.curiosity * 0.25)
        self.social_affiliation += self.update_rate * (
            _clip(0.5 + 0.5 * event.social_feedback) - self.social_affiliation
        )
        if event.correction:
            self.uncertainty = _clip(self.uncertainty + 0.1)
            self.competence = _clip(self.competence - 0.05)
        self._bound_state()

    def modulation(self) -> float:
        """Return the global third-factor multiplier for plasticity."""

        return _clip(
            0.5 + 0.5 * self.uncertainty + 0.25 * self.curiosity + 0.25 * self.arousal, 0.1, 2.0
        )

    def derived_labels(self) -> dict[str, float]:
        """Return inspectable, non-primitive affect labels."""

        return {
            "happy_like": _clip(
                0.5 + 0.5 * self.valence + 0.25 * self.competence - 0.25 * self.threat
            ),
            "alarmed_like": _clip(0.5 * self.threat + 0.5 * self.arousal),
            "exploratory_like": _clip(
                self.curiosity * (1.0 - self.threat) * (0.5 + 0.5 * self.arousal)
            ),
            "uncertain_like": _clip(0.5 * self.uncertainty + 0.5 * (1.0 - self.competence)),
        }

    def as_dict(self) -> dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "uncertainty": self.uncertainty,
            "curiosity": self.curiosity,
            "threat": self.threat,
            "competence": self.competence,
            "social_affiliation": self.social_affiliation,
            "energy": self.energy,
            **self.derived_labels(),
        }
