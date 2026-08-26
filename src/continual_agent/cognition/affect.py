"""Bounded, inspectable affective and motivational state."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    _baselines: dict[str, float] = field(
        default_factory=lambda: {
            "valence": 0.0,
            "arousal": 0.2,
            "uncertainty": 0.5,
            "curiosity": 0.5,
            "threat": 0.1,
            "competence": 0.5,
            "social_affiliation": 0.5,
            "energy": 1.0,
        },
        repr=False,
    )

    def __post_init__(self) -> None:
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
        for _ in range(steps):
            for name, baseline in self._baselines.items():
                if name == "energy":
                    continue
                value = getattr(self, name)
                setattr(self, name, value + self.decay_rate * (baseline - value))
            self.energy = _clip(self.energy - 0.002)
        self._bound_state()

    def recover(self, steps: int = 1) -> None:
        """Recover energy during idle time without changing other state."""

        if steps < 0:
            raise ValueError("steps cannot be negative")
        self.energy = _clip(self.energy + 0.01 * steps)

    def observe(self, event: AffectiveEvent) -> None:
        """Update state from an event and its delayed outcome."""

        rpe = max(-1.0, min(1.0, event.reward_prediction_error))
        threat = _clip(event.threat)
        urgency = _clip(event.urgency)
        novelty = _clip(event.novelty)
        progress = _clip(event.learning_progress)
        uncertainty = _clip(event.uncertainty)

        self.valence += self.update_rate * rpe
        self.competence += self.update_rate * (rpe if rpe >= 0 else 0.5 * rpe)
        self.threat += self.update_rate * (threat - self.threat)
        self.arousal += self.update_rate * (
            max(urgency, threat) - self.arousal
        )
        self.uncertainty += self.update_rate * (uncertainty - self.uncertainty)
        self.curiosity += self.update_rate * (
            novelty * progress - self.curiosity * 0.25
        )
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
            0.5
            + 0.5 * self.uncertainty
            + 0.25 * self.curiosity
            + 0.25 * self.arousal
        , 0.1, 2.0)

    def derived_labels(self) -> dict[str, float]:
        """Return inspectable, non-primitive affect labels."""

        return {
            "happy_like": _clip(0.5 + 0.5 * self.valence + 0.25 * self.competence - 0.25 * self.threat),
            "alarmed_like": _clip(0.5 * self.threat + 0.5 * self.arousal),
            "exploratory_like": _clip(self.curiosity * (1.0 - self.threat) * (0.5 + 0.5 * self.arousal)),
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
