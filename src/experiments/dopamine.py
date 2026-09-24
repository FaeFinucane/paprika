"""Fixed cue-to-outcome conditioning runs for the VTA circuit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..builder import NetworkBuilder, PopulationHandle
from ..circuits.attractor import AttractorHandle, AttractorSpec, add_attractor
from ..circuits.synfire import SynfireChainHandle, SynfireChainSpec, add_synfire_chain
from ..circuits.vta_dopamine import (
    VtaDopamineHandle,
    VtaDopamineSystem,
    add_vta_dopamine,
    compiled_vta_dopamine,
)
from ..interaction.rates import (
    PopulationRate,
    PopulationRateSpec,
    RatePatternInput,
    UnipolarRateInput,
)
from ..network.connectivity import FanInSpec, StrengthSpec
from ..network.population import Population
from ..session import Session
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
    response_tail_ticks: int = 5
    inter_trial_ticks: int = 30
    cue_pattern: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if min(self.settle_ticks, self.response_tail_ticks, self.inter_trial_ticks) < 0:
            raise ValueError("settle, response-tail, and inter-trial ticks must be non-negative")
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
        """The event-aligned cue/delay/outcome/response window per trial."""
        return (
            self.cue_ticks
            + self.cue_to_outcome_ticks
            + self.outcome_ticks
            + self.response_tail_ticks
        )


@dataclass
class ConditioningRun:
    """One seed's compiled circuit and event-aligned observations."""

    seed: int
    circuit: "FixedCueOutcomeConditioningCircuit"
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

    def population_rate(self, population: str | PopulationHandle) -> np.ndarray:
        """Return a requested population's per-tick instantaneous rate stream."""
        name = population.name if isinstance(population, PopulationHandle) else population
        return self.stream(f"rate:{name}")

    def snapshot(self, name: str) -> np.ndarray:
        """Stack one snapshot source as ``[seed, boundary, ...]``."""
        if not self.runs:
            raise ValueError("report has no runs")
        try:
            return np.stack([run.snapshots[name] for run in self.runs])
        except KeyError as error:
            raise KeyError(f"unknown snapshot {name!r}") from error


@dataclass
class FixedCueOutcomeConditioningCircuit:
    """Runtime state of the fixed cue-to-outcome conditioning experiment."""

    session: Session
    populations: dict[str, Population]
    cue_input: RatePatternInput
    outcome_positive: UnipolarRateInput
    outcome_negative: UnipolarRateInput
    inferred_state: AttractorHandle
    temporal: SynfireChainHandle
    vta: VtaDopamineSystem

    @property
    def dopamine_rate(self) -> PopulationRate:
        return self.vta.dopamine_rate

    @property
    def dopamine(self):
        return self.vta.dopamine

    @property
    def stdp(self):
        return self.vta.stdp

    def present_cue(self, pattern: np.ndarray, duration: int | None = None) -> None:
        """Present a bounded unipolar cue without clearing ongoing network state."""
        self.cue_input.write(pattern, duration)

    def deliver_outcome(self, value: float, duration: int | None = None) -> None:
        """Route one signed outcome through this experiment's sensory pathways."""
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("outcome must be finite and in [-1, 1]")
        if value >= 0:
            self.outcome_positive.write(float(value), duration)
        else:
            self.outcome_negative.write(float(-value), duration)

    def tick(self):
        return self.session.tick()


@dataclass(frozen=True, slots=True)
class FixedCueOutcomeConditioningDeclaration:
    """Parent assembly for the original fixed cue-to-outcome experiment."""

    builder: NetworkBuilder
    cue: PopulationHandle
    outcome_positive: PopulationHandle
    outcome_negative: PopulationHandle
    inferred_state: AttractorHandle
    temporal: SynfireChainHandle
    vta: VtaDopamineHandle

    def build(
        self, seed: int = 0, *, enable_dopamine_learning: bool = True
    ) -> FixedCueOutcomeConditioningCircuit:
        """Compile the assembly and initialise its experiment-local inputs."""
        session = self.builder.compile(seed)
        populations = {
            population.spec.name: population for population in session.snn.layout.populations
        }
        vta = compiled_vta_dopamine(session, self.vta)
        if not enable_dopamine_learning:
            session.remove(vta.stdp)
        cue_input = RatePatternInput(populations[self.cue.name], np.random.default_rng([seed, 1]))
        outcome_positive = UnipolarRateInput(
            populations[self.outcome_positive.name], np.random.default_rng([seed, 2])
        )
        outcome_negative = UnipolarRateInput(
            populations[self.outcome_negative.name], np.random.default_rng([seed, 3])
        )
        session.add(cue_input, outcome_positive, outcome_negative)
        return FixedCueOutcomeConditioningCircuit(
            session,
            populations,
            cue_input,
            outcome_positive,
            outcome_negative,
            self.inferred_state,
            self.temporal,
            vta,
        )


def add_fixed_cue_outcome_conditioning(
    builder: NetworkBuilder,
    *,
    attractor: AttractorSpec = AttractorSpec(),
    temporal: SynfireChainSpec = SynfireChainSpec(),
) -> FixedCueOutcomeConditioningDeclaration:
    """Compose the original cue/reward experiment from reusable circuits.

    This function owns every cross-circuit projection.  The attractor,
    temporal basis, and VTA component each own only their local dynamics.
    """
    cue = builder.add_feature_population("CUE", ("CUE",), width=16)
    outcome_positive = builder.add_population("OUTCOME_POSITIVE", 16)
    outcome_negative = builder.add_population("OUTCOME_NEGATIVE", 16, output="inhibitory")
    inferred_state = add_attractor(builder, attractor)
    timer = add_synfire_chain(builder, temporal)
    vta = add_vta_dopamine(builder)

    builder.connect(
        cue,
        inferred_state.input,
        FanInSpec(6),
        StrengthSpec(0.25, 0.02, maximum=0.8),
        "dopamine_stdp",
        True,
        f"{cue.name}_to_{inferred_state.input}",
    )
    builder.connect(
        inferred_state.output,
        timer.input,
        FanInSpec(4),
        StrengthSpec(0.30, 0.02, maximum=0.8),
        "fixed",
        False,
        f"{inferred_state.output}_to_{timer.input}",
    )
    builder.connect(
        inferred_state.output,
        vta.dopamine_input,
        FanInSpec(min(8, attractor.excitatory_size)),
        StrengthSpec(0.12, 0.02, maximum=0.8),
        "dopamine_stdp",
        True,
        f"{inferred_state.output}_to_{vta.dopamine_input}",
    )
    for source in timer.outputs:
        builder.connect(
            source,
            vta.inhibitory_input,
            FanInSpec(min(8, temporal.stage_size)),
            StrengthSpec(0.10, 0.02, maximum=0.8),
            "dopamine_stdp",
            True,
            f"{source}_to_{vta.inhibitory_input}",
        )
    for outcome in (outcome_positive, outcome_negative):
        builder.connect(
            outcome,
            vta.dopamine_input,
            FanInSpec(8),
            StrengthSpec(0.30, 0.02, maximum=0.8),
            "fixed",
            False,
            f"{outcome.name}_to_{vta.dopamine_input}",
        )

    return FixedCueOutcomeConditioningDeclaration(
        builder,
        cue,
        outcome_positive,
        outcome_negative,
        inferred_state,
        timer,
        vta,
    )


def run_fixed_cue_outcome(
    declaration: FixedCueOutcomeConditioningDeclaration,
    protocol: FixedCueOutcomeProtocol = FixedCueOutcomeProtocol(),
    *,
    seeds: Iterable[int] = (0,),
    population_rates: Iterable[str | PopulationHandle] = (),
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
    declaration: FixedCueOutcomeConditioningDeclaration,
    protocol: FixedCueOutcomeProtocol,
    seed: int,
    requested_rates: tuple[str | PopulationHandle, ...],
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
        _tick(circuit, protocol.outcome_ticks + protocol.response_tail_ticks)
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
    circuit: FixedCueOutcomeConditioningCircuit,
    requested_rates: tuple[str | PopulationHandle, ...],
) -> None:
    existing = {
        observer.population.spec.name
        for observer in circuit.session.observers
        if isinstance(observer, PopulationRate)
    }
    names = [
        handle.name if isinstance(handle, PopulationHandle) else handle
        for handle in requested_rates
    ]
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise ValueError(f"population rates requested more than once: {', '.join(duplicate)}")
    for name in names:
        if name in existing:
            raise ValueError(f"a PopulationRate already observes {name!r}")
        try:
            population = circuit.populations[name]
        except KeyError as error:
            raise KeyError(f"unknown declared population {name!r}") from error
        circuit.session.add(PopulationRate(population, PopulationRateSpec(decay=0.0)))


def _capture_snapshots(
    circuit: FixedCueOutcomeConditioningCircuit,
    sources: tuple[SnapshotSource, ...],
    captured: dict[str, list[np.ndarray]],
) -> None:
    for source in sources:
        captured[source.report_name].append(np.asarray(source.snapshot(circuit.session.snn)).copy())


def _cue_pattern(
    circuit: FixedCueOutcomeConditioningCircuit, protocol: FixedCueOutcomeProtocol
) -> np.ndarray:
    if protocol.cue_pattern is None:
        pattern = np.zeros(circuit.cue_input.population.count)
        pattern[: pattern.size // 2] = 1.0
        return pattern
    if protocol.cue_pattern.shape != (circuit.cue_input.population.count,):
        raise ValueError("cue_pattern must match the CUE population")
    return protocol.cue_pattern.copy()


def _tick(circuit: FixedCueOutcomeConditioningCircuit, ticks: int) -> None:
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
    "FixedCueOutcomeConditioningCircuit",
    "FixedCueOutcomeConditioningDeclaration",
    "FixedCueOutcomeProtocol",
    "add_fixed_cue_outcome_conditioning",
    "run_fixed_cue_outcome",
]
