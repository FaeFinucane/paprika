"""Population readout and adaptive deliberation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np

from continual_agent.simulation.population_layout import Population, PopulationLayout


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


@dataclass(frozen=True)
class OutputEvent:
    """One discrete event from a named output subgroup."""

    population: Population
    name: str
    timestamp: int
    evidence: float
    interval: int | None = None

    @property
    def is_eos(self) -> bool:
        return self.population is Population.OUTPUT_CHAR and self.name == "<EOS>"

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class OutputCandidate:
    """Evidence for one output subgroup in a single observation."""

    population: Population
    name: str
    evidence: float
    order: int

    @property
    def is_eos(self) -> bool:
        return self.population is Population.OUTPUT_CHAR and self.name == "<EOS>"


@dataclass(frozen=True)
class ArbitrationDecision:
    """Inspectable result of one independent output arbitration step."""

    selected: OutputCandidate | None
    candidates: tuple[OutputCandidate, ...]
    reason: str


class OutputArbitrationPolicy:
    """Deterministically select at most one candidate event.

    Evidence is the primary signal.  Priorities only resolve equal evidence:
    EOS wins over a character, actions win over characters, and subgroup order
    is the final stable tie-breaker. Activation state is deliberately host-side
    state, supplied by ``EventReadout``.
    """

    def __init__(
        self,
        *,
        action_priority: int = 1,
        character_priority: int = 0,
        eos_priority: int = 2,
    ) -> None:
        self.action_priority = action_priority
        self.character_priority = character_priority
        self.eos_priority = eos_priority

    def arbitrate(
        self,
        candidates: Iterable[OutputCandidate],
    ) -> ArbitrationDecision:
        candidates = tuple(candidates)
        if not candidates:
            return ArbitrationDecision(None, candidates, "no_candidates")

        def rank(candidate: OutputCandidate) -> tuple[float, int, int]:
            if candidate.is_eos:
                priority = self.eos_priority
            elif candidate.population is Population.OUTPUT_ACTION:
                priority = self.action_priority
            else:
                priority = self.character_priority
            # Lower subgroup order is stable and wins a complete tie.
            return (candidate.evidence, priority, -candidate.order,)

        selected = max(candidates, key=rank)
        return ArbitrationDecision(selected, candidates, "selected")


class EventReadout:
    """Convert output-population evidence into timestamped discrete events."""

    def __init__(
        self,
        layout: PopulationLayout,
        *,
        activation_threshold: float = 1.0,
        release_threshold: float = 0.5,
        arbitration: OutputArbitrationPolicy | None = None,
    ) -> None:
        if activation_threshold <= 0 or release_threshold <= 0:
            raise ValueError("activation and release thresholds must be positive")
        if release_threshold >= activation_threshold:
            raise ValueError("release_threshold must be less than activation_threshold")
        self.layout = layout
        self.activation_threshold = float(activation_threshold)
        self.release_threshold = float(release_threshold)
        self.arbitration = arbitration or OutputArbitrationPolicy()
        self._groups: dict[Population, dict[str, np.ndarray]] = {
            Population.OUTPUT_ACTION: self._named_groups(layout.action_subgroups),
            Population.OUTPUT_CHAR: self._named_groups(layout.char_subgroups),
        }
        if not any(self._groups.values()):
            raise ValueError("layout must define output action or character subgroups")
        self.reset()

    @staticmethod
    def _named_groups(groups: Mapping[str, slice]) -> dict[str, np.ndarray]:
        return {name: np.arange(bounds.start, bounds.stop) for name, bounds in groups.items()}

    def reset(self) -> None:
        self.timestamp = -1
        self.stopped = False
        self.events: list[OutputEvent] = []
        self.arbitrations: list[ArbitrationDecision] = []
        self.last_arbitration: ArbitrationDecision | None = None
        self.active_output: tuple[Population, str] | None = None
        self._last_event_timestamp: int | None = None

    def observe(
        self,
        frame: np.ndarray,
        *,
        activation: np.ndarray | None = None,
        activations: np.ndarray | None = None,
        timestamp: int | None = None,
        populations: Iterable[Population] | None = None,
    ) -> OutputEvent | None:
        """Observe one tick; ``None`` means silence, never EOS.

        ``frame`` can be spikes.  ``activation``/``activations`` optionally
        supplies a continuous activation trace; the stronger evidence at each
        neuron is used without counting a spike and activation twice.
        """
        values = np.asarray(frame, dtype=float)
        if values.ndim != 1 or values.size < self.layout.total_count:
            raise ValueError("output frame must cover the named network layout")
        if activation is not None and activations is not None:
            raise TypeError("provide either activation or activations, not both")
        activation_values = activation if activation is not None else activations
        if activation_values is not None:
            activation_values = np.asarray(activation_values, dtype=float)
            if activation_values.ndim != 1 or activation_values.size < self.layout.total_count:
                raise ValueError("activation frame must cover the named network layout")
            values = np.maximum(values, activation_values)
        if timestamp is None:
            timestamp = self.timestamp + 1
        if timestamp <= self.timestamp:
            raise ValueError("timestamps must increase")
        self.timestamp = timestamp
        if self.stopped:
            return None
        if self.active_output is not None:
            population, name = self.active_output
            group = self._groups[population][name]
            if float(values[group].sum()) >= self.release_threshold:
                self.last_arbitration = ArbitrationDecision(None, (), "active")
                self.arbitrations.append(self.last_arbitration)
                return None
            self.active_output = None
        selected = tuple(populations) if populations is not None else tuple(self._groups)
        candidates: list[OutputCandidate] = []
        for population in selected:
            for order, (name, group) in enumerate(self._groups.get(population, {}).items()):
                evidence = float(values[group].sum())
                if evidence >= self.activation_threshold:
                    candidates.append(OutputCandidate(population, name, evidence, order))
        decision = self.arbitration.arbitrate(candidates)
        self.last_arbitration = decision
        self.arbitrations.append(decision)
        if decision.selected is None:
            return None
        selected = decision.selected
        evidence, name, population = selected.evidence, selected.name, selected.population
        key = (population, name)
        self.active_output = key
        event = OutputEvent(
            population,
            name,
            timestamp,
            evidence,
            None if self._last_event_timestamp is None else timestamp - self._last_event_timestamp,
        )
        self.events.append(event)
        self._last_event_timestamp = timestamp
        if event.is_eos:
            self.stopped = True
        return event

class ActionReadout:
    """Expose named action population bounds for neural training/debugging."""
    def __init__(
        self,
        actions: tuple[Action, ...] = tuple(Action),
        neurons_per_action: int = 4,
    ):
        if not actions or neurons_per_action <= 0:
            raise ValueError("actions and neurons_per_action must be non-empty")
        self.actions = actions
        self.neurons_per_action = neurons_per_action

    @property
    def neuron_count(self) -> int:
        return len(self.actions) * self.neurons_per_action

    def groups(self, layout: PopulationLayout) -> dict[Action, np.ndarray]:
        """Return the layout's named action subgroups."""
        bounds = layout.slice(Population.OUTPUT_ACTION)
        if bounds.stop - bounds.start != self.neuron_count:
            raise ValueError("layout output_action population does not match readout")
        try:
            return {
                action: np.arange(
                    layout.subgroup(Population.OUTPUT_ACTION, action.value).start,
                    layout.subgroup(Population.OUTPUT_ACTION, action.value).stop,
                )
                for action in self.actions
            }
        except KeyError as error:
            raise ValueError("layout must define every action subgroup") from error
