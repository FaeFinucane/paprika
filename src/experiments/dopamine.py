"""Fixed cue-to-outcome conditioning runs for the VTA circuit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..builder import PopulationHandle
from ..circuits.vta_dopamine import DopamineCircuit, VtaDopamineCircuitDeclaration
from ..interaction.rates import PopulationRate
from .recording import SignalRecorder, SnapshotSource


@dataclass(frozen=True)
class FixedCueOutcomeProtocol:
    """One repeatable cue, delay, outcome, and inter-trial schedule."""

    trials: int = 100
    settle_ticks: int = 1_000
    cue_ticks: int = 5
    cue_to_outcome_ticks: int = 5
    outcome: float = 0.7
    outcome_ticks: int = 10
    inter_trial_ticks: int = 30
    cue_pattern: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if self.settle_ticks < 0 or self.inter_trial_ticks < 0:
            raise ValueError("settle and inter-trial ticks must be non-negative")
        if self.cue_ticks <= 0 or self.cue_to_outcome_ticks < 0 or self.outcome_ticks <= 0:
            raise ValueError("invalid cue-to-outcome timing")
        if not np.isfinite(self.outcome) or not -1.0 <= self.outcome <= 1.0:
            raise ValueError("outcome must be finite and in [-1, 1]")
        if self.cue_pattern is not None:
            pattern = np.asarray(self.cue_pattern, dtype=float)
            if pattern.ndim != 1 or not np.all(np.isfinite(pattern)):
                raise ValueError("cue_pattern must be a finite one-dimensional array")
            if np.any(pattern < 0.0) or np.any(pattern > 1.0):
                raise ValueError("cue_pattern rates must be in [0, 1]")
            object.__setattr__(self, "cue_pattern", pattern.copy())

    @property
    def event_ticks(self) -> int:
        """The event-aligned cue/delay/outcome window recorded for each trial."""
        return self.cue_ticks + self.cue_to_outcome_ticks + self.outcome_ticks


@dataclass
class ConditioningRun:
    """One seed's compiled circuit and event-aligned observations."""

    seed: int
    circuit: DopamineCircuit
    tick_streams: dict[str, np.ndarray]
    snapshots: dict[str, np.ndarray]


@dataclass
class ConditioningReport:
    """Uniform access to one or more independent conditioning runs."""

    protocol: FixedCueOutcomeProtocol
    runs: tuple[ConditioningRun, ...]

    @property
    def report_names(self) -> tuple[str, ...]:
        return tuple(self.runs[0].tick_streams) if self.runs else ()

    @property
    def snapshot_names(self) -> tuple[str, ...]:
        return tuple(self.runs[0].snapshots) if self.runs else ()

    @property
    def dopamine(self) -> np.ndarray:
        """Dopamine per ``[seed, trial, event_tick]``."""
        return self.stream("dopamine")

    def stream(self, name: str) -> np.ndarray:
        """Stack one tick stream as ``[seed, trial, event_tick, ...]``."""
        if not self.runs:
            raise ValueError("report has no runs")
        try:
            return np.stack([run.tick_streams[name] for run in self.runs])
        except KeyError as error:
            raise KeyError(f"unknown tick stream {name!r}") from error

    def population_rate(self, population: PopulationHandle) -> np.ndarray:
        """Return a requested population's per-tick instantaneous rate stream."""
        return self.stream(f"rate:{population.name}")

    def snapshot(self, name: str) -> np.ndarray:
        """Stack one snapshot source as ``[seed, boundary, ...]``."""
        if not self.runs:
            raise ValueError("report has no runs")
        try:
            return np.stack([run.snapshots[name] for run in self.runs])
        except KeyError as error:
            raise KeyError(f"unknown snapshot {name!r}") from error


def run_fixed_cue_outcome(
    declaration: VtaDopamineCircuitDeclaration,
    protocol: FixedCueOutcomeProtocol = FixedCueOutcomeProtocol(),
    *,
    seeds: Iterable[int] = (0,),
    population_rates: Iterable[PopulationHandle] = (),
    snapshots: Iterable[SnapshotSource] = (),
    enable_dopamine_learning: bool = True,
) -> ConditioningReport:
    """Run an event-aligned fixed conditioning protocol for one or more seeds.

    Registered ``PopulationRate`` and ``DopamineReadout`` values are discovered
    automatically. ``population_rates`` adds instantaneous (decay-zero) rate
    observers for additional declared populations. Snapshot sources are
    captured after settling, after every trial, and once after the final
    inter-trial interval.
    """
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")

    requested_rates = tuple(population_rates)
    snapshot_sources = tuple(snapshots)
    _validate_snapshot_names(snapshot_sources)
    runs = tuple(
        _run_seed(
            declaration,
            protocol,
            seed,
            requested_rates,
            snapshot_sources,
            enable_dopamine_learning,
        )
        for seed in seed_values
    )
    return ConditioningReport(protocol, runs)


def _run_seed(
    declaration: VtaDopamineCircuitDeclaration,
    protocol: FixedCueOutcomeProtocol,
    seed: int,
    requested_rates: tuple[PopulationHandle, ...],
    snapshot_sources: tuple[SnapshotSource, ...],
    enable_dopamine_learning: bool,
) -> ConditioningRun:
    circuit = declaration.build(seed, enable_dopamine_learning=enable_dopamine_learning)
    _add_requested_rates(circuit, requested_rates)
    _tick(circuit, protocol.settle_ticks)
    recorder = SignalRecorder.from_session(circuit.session)
    circuit.session.add(recorder)

    pattern = _cue_pattern(circuit, protocol)
    event_streams: dict[str, list[np.ndarray]] = {name: [] for name in recorder.report_names}
    captured: dict[str, list[np.ndarray]] = {source.report_name: [] for source in snapshot_sources}
    _capture_snapshots(circuit, snapshot_sources, captured)

    for _ in range(protocol.trials):
        start = recorder.sample_count
        circuit.present_cue(pattern, protocol.cue_ticks)
        _tick(circuit, protocol.cue_ticks + protocol.cue_to_outcome_ticks)
        circuit.deliver_outcome(protocol.outcome, protocol.outcome_ticks)
        _tick(circuit, protocol.outcome_ticks)
        for name, samples in recorder.samples_since(start).items():
            event_streams[name].append(samples)
        _capture_snapshots(circuit, snapshot_sources, captured)
        _tick(circuit, protocol.inter_trial_ticks)

    _capture_snapshots(circuit, snapshot_sources, captured)
    return ConditioningRun(
        seed,
        circuit,
        {name: np.stack(samples) for name, samples in event_streams.items()},
        {name: np.stack(samples) for name, samples in captured.items()},
    )


def _add_requested_rates(
    circuit: DopamineCircuit, requested_rates: tuple[PopulationHandle, ...]
) -> None:
    existing = {
        observer.population.spec.name
        for observer in circuit.session.observers
        if isinstance(observer, PopulationRate)
    }
    names = [handle.name for handle in requested_rates]
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise ValueError(f"population rates requested more than once: {', '.join(duplicate)}")
    for handle in requested_rates:
        if handle.name in existing:
            raise ValueError(f"a PopulationRate already observes {handle.name!r}")
        try:
            population = circuit.populations[handle.name]
        except KeyError as error:
            raise KeyError(f"unknown declared population {handle.name!r}") from error
        circuit.session.add(PopulationRate(population, decay=0.0))


def _capture_snapshots(
    circuit: DopamineCircuit,
    sources: tuple[SnapshotSource, ...],
    captured: dict[str, list[np.ndarray]],
) -> None:
    for source in sources:
        captured[source.report_name].append(np.asarray(source.snapshot(circuit.session.snn)).copy())


def _cue_pattern(circuit: DopamineCircuit, protocol: FixedCueOutcomeProtocol) -> np.ndarray:
    if protocol.cue_pattern is None:
        pattern = np.zeros(circuit.cue_input.population.count)
        pattern[: pattern.size // 2] = 1.0
        return pattern
    if protocol.cue_pattern.shape != (circuit.cue_input.population.count,):
        raise ValueError("cue_pattern must match the CUE population")
    return protocol.cue_pattern.copy()


def _tick(circuit: DopamineCircuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()


def _validate_snapshot_names(sources: tuple[SnapshotSource, ...]) -> None:
    names = [source.report_name for source in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate snapshot names: {', '.join(duplicates)}")


__all__ = [
    "ConditioningReport",
    "ConditioningRun",
    "FixedCueOutcomeProtocol",
    "run_fixed_cue_outcome",
]
