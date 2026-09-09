"""Generic per-trial record shared across experiments, plus a stateful
recorder that builds one. Cue ticks are stored relative to the trial's own
start, so trials can be compared/aligned regardless of where in the
simulation they occurred.

Experiments own *when* a cue occurs and what an extra means (a response
token, a reward value, ...) - they call start/mark/finalize at the right
points, passing the current tick explicitly (a trial's boundaries are
defined by the experiment's control flow, not by spikes, so there's no need
to observe the session to know "now"). TrialRecorder alone owns building
the Trial: a Diagnostic never touches it directly, it just returns the
extras it wants recorded from finalize() and TrialRecorder merges them in.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Cue(Enum):
    BASELINE = auto()
    PROMPT = auto()
    RESPONSE = auto()
    REWARD = auto()
    PUNISH = auto()
    CANCELLED = auto()


@dataclass
class Trial:
    start_tick: int
    duration: int
    cues: list[tuple[Cue, int]] = field(default_factory=list[tuple[Cue, int]])
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    def cue_tick(self, cue: Cue) -> int | None:
        """Tick (relative to start_tick) of the first mark of `cue`, if any."""
        for marked, tick in self.cues:
            if marked is cue:
                return tick
        return None

    def has_cue(self, cue: Cue) -> bool:
        return self.cue_tick(cue) is not None

    def count_cue(self, cue: Cue) -> int:
        return sum(1 for marked, _ in self.cues if marked is cue)


class Diagnostic(ABC):
    """A plugin TrialRecorder notifies as a trial progresses. A Diagnostic
    keeps its own working state (wiping it in start(), typically) and hands
    back whatever it wants recorded from finalize() - it never reaches into
    the Trial or TrialRecorder itself."""

    def start(self) -> None:
        pass

    def mark(self, _cue: Cue, _tick: int) -> None:
        pass

    def finalize(self) -> dict[str, Any]:
        return {}


@dataclass
class TrialRecorder:
    diagnostics: list[Diagnostic] = field(default_factory=list[Diagnostic])

    trials: list[Trial] = field(init=False, default_factory=list[Trial])
    _start_tick: int = field(init=False, default=0)
    _cues: list[tuple[Cue, int]] = field(init=False, default_factory=list[tuple[Cue, int]])
    _extras: dict[str, Any] = field(init=False, default_factory=dict[str, Any])

    def start(self, tick: int):
        self._start_tick = tick
        self._cues = []
        self._extras = {}
        for diagnostic in self.diagnostics:
            diagnostic.start()

    def mark(self, cue: Cue, tick: int):
        rel_tick = tick - self._start_tick
        self._cues.append((cue, rel_tick))
        for diagnostic in self.diagnostics:
            diagnostic.mark(cue, rel_tick)

    def set_extra(self, key: str, value: Any):
        self._extras[key] = value

    def finalize(self, tick: int) -> Trial:
        extras = dict(self._extras)
        for diagnostic in self.diagnostics:
            extras.update(diagnostic.finalize())
        trial = Trial(self._start_tick, tick - self._start_tick, list(self._cues), extras)
        self.trials.append(trial)
        return trial
