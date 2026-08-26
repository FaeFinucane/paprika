"""Population readout and adaptive deliberation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Action(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"
    UNCERTAIN = "uncertain"
    REVISE = "revise"
    REFUSE = "refuse"
    WAIT = "wait"


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    ticks: int
    evidence: dict[Action, float]
    timed_out: bool


class ActionReadout:
    def __init__(
        self,
        actions: tuple[Action, ...] = tuple(Action),
        neurons_per_action: int = 4,
        threshold: float = 2.0,
        margin: float = 0.25,
    ):
        if not actions or neurons_per_action <= 0:
            raise ValueError("actions and neurons_per_action must be non-empty")
        self.actions = actions
        self.neurons_per_action = neurons_per_action
        self.threshold = threshold
        self.margin = margin

    def score_policy(self, features: np.ndarray) -> dict[Action, float]:
        """Return optional policy scores for a compact local readout.

        The first vertical slice uses this as a transparent three-factor
        readout: features are presynaptic activity, the chosen action is the
        postsynaptic population, and reward is the third factor.
        """

        features = np.asarray(features, dtype=float)
        if not hasattr(self, "policy_weights"):
            self.policy_weights = np.zeros((len(self.actions), features.size))
        if self.policy_weights.shape[1] != features.size:
            self.policy_weights = np.zeros((len(self.actions), features.size))
        return {
            action: float(self.policy_weights[index] @ features)
            for index, action in enumerate(self.actions)
        }

    def reinforce_policy(
        self, action: Action, features: np.ndarray, reward_error: float, learning_rate: float = 0.1
    ) -> None:
        features = np.asarray(features, dtype=float)
        self.score_policy(features)
        index = self.actions.index(action)
        self.policy_weights[index] += learning_rate * reward_error * features
        self.policy_weights[index] = np.clip(self.policy_weights[index], -2.0, 2.0)

    @property
    def neuron_count(self) -> int:
        return len(self.actions) * self.neurons_per_action

    def groups(self, start: int) -> dict[Action, np.ndarray]:
        return {
            action: np.arange(
                start + index * self.neurons_per_action,
                start + (index + 1) * self.neurons_per_action,
            )
            for index, action in enumerate(self.actions)
        }

    def decide(self, spikes: list[np.ndarray], start: int) -> Decision:
        groups = self.groups(start)
        evidence = {
            action: float(sum(frame[group].sum() for frame in spikes))
            for action, group in groups.items()
        }
        ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)
        best_action, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = best / max(1.0, sum(evidence.values()))
        ready = best >= self.threshold and (best - second) >= self.margin
        return Decision(
            action=best_action,
            confidence=confidence,
            ticks=len(spikes),
            evidence=evidence,
            timed_out=not ready,
        )
