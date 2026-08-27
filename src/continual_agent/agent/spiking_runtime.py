"""Canonical composition root and efficient hot-path loop for the SNN."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock
from typing import TypeVar

import numpy as np

from continual_agent.agent.drives import ArrayDrive, BackgroundDrive
from continual_agent.agent.input_runner import InputRunner
from continual_agent.agent.network_config import NetworkConfig
from continual_agent.agent.plugins import (
    HomeostasisPlugin,
    MetricsPlugin,
    NetworkContext,
    NetworkPlugin,
    PlasticityPlugin,
)
from continual_agent.agent.population_homeostasis import PopulationHomeostasis
from continual_agent.agent.runtime_metrics import RuntimeMetrics
from continual_agent.agent.runtime_session import RuntimeSession
from continual_agent.agent.session import (
    ConflictPolicy,
    InputSignal,
    ResponseSession,
    SessionExecutionSnapshot,
)
from continual_agent.cognition.readout import EventReadout, OutputEvent
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.core import NetworkCore
from continual_agent.simulation.neurons import LIFNeurons
from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses
from continual_agent.simulation.weight_initialization import WeightInitializer

T = TypeVar("T")


class SpikingRuntime:
    """Compose network services while keeping one vectorised tick loop."""

    # PopulationLayout should be the only record for neuron sizes.
    layout: PopulationLayout
    network: NetworkCore
    output_readout: EventReadout
    edge_enabled: np.ndarray
    plasticity: RewardModulatedSTDP
    response_session: ResponseSession
    metrics: RuntimeMetrics
    metrics_plugin: MetricsPlugin
    # Very confused - drives are present twice in these properties? Why
    homeostasis: PopulationHomeostasis
    external_drive: ArrayDrive
    background_drive: BackgroundDrive
    plugins: list[NetworkPlugin]
    _context: NetworkContext
    hidden_feature_groups: tuple[np.ndarray, ...]
    direct_input_output_edge_indices: np.ndarray
    hidden_output_edge_indices: np.ndarray
    hidden_recurrent_edge_indices: np.ndarray
    token_input_edge_indices: np.ndarray
    affect_edge_indices: np.ndarray
    affect_action_edge_indices: np.ndarray
    trainer: InputRunner

    def __init__(self, config: NetworkConfig) -> None:
        self.layout = config.layout
        input_features = self.layout.input_count
        affect_bounds = self.layout.slice(Population.AFFECT)
        action_bounds = self.layout.slice(Population.OUTPUT_ACTION)
        char_bounds = self.layout.slice(Population.OUTPUT_CHAR)
        total = self.layout.total_count
        output_tokens = tuple(self.layout.char_subgroups)
        affect_start, action_start, char_start = (
            affect_bounds.start,
            action_bounds.start,
            char_bounds.start,
        )
        neurons = LIFNeurons(total, dt=1.0, tau_membrane=5.0, threshold=1.0, refractory_ticks=2)
        initializer = WeightInitializer(config.seed, config.weight_initialization)
        c = initializer.config
        synapses = initializer.random_synapses(total, config.connection_probability)
        explicit_pairs = np.concatenate(
            (
                *(
                    np.stack(np.meshgrid(sources, targets, indexing="ij"), axis=-1).reshape(-1, 2)
                    for sources, targets in (
                        (
                            np.arange(input_features),
                            np.arange(input_features, affect_start),
                        ),
                        (np.arange(input_features), np.arange(affect_start, action_start)),
                        (np.arange(input_features), np.arange(action_start, char_start)),
                        (np.arange(input_features), np.arange(char_start, total)),
                        (
                            np.arange(input_features, affect_start),
                            np.arange(char_start, total),
                        ),
                        (
                            np.arange(affect_start, action_start),
                            np.arange(action_start, char_start),
                        ),
                    )
                ),
            )
        )
        explicit = set(map(tuple, explicit_pairs.tolist()))
        valid = np.array(
            [
                (int(source), int(target)) not in explicit and target < char_start
                for source, target in zip(synapses.source, synapses.target)
            ],
            dtype=bool,
        )
        synapses = SparseSynapses(
            synapses.source[valid], synapses.target[valid], synapses.weight[valid], total
        )
        hidden_groups = tuple(
            np.asarray(g, dtype=np.int64)
            for g in np.array_split(np.arange(input_features, affect_start), input_features)
        )
        for source, group in enumerate(hidden_groups):
            initializer.bootstrap_input_hidden(
                synapses, np.array([source]), group, f"input_hidden_bootstrap_{source}"
            )
        for index, seed in enumerate(c.population_projections):
            if seed.source is Population.OUTPUT_CHAR:
                raise ValueError("OUTPUT_CHAR cannot be a projection source")
            source_bounds = self.layout.slice(seed.source)
            target_bounds = self.layout.slice(seed.target)
            for contact in range(seed.contacts):
                initializer.population_projection(
                    synapses,
                    seed,
                    np.arange(source_bounds.start, source_bounds.stop),
                    np.arange(target_bounds.start, target_bounds.stop),
                    f"population_projection_{index}_{contact}",
                )

        def projection(
            name: str, sources: np.ndarray, targets: np.ndarray, mean: float, spread: float
        ) -> np.ndarray:
            return initializer.projection(synapses, name, sources, targets, mean, spread)

        projection(
            "input_action",
            np.arange(input_features),
            np.arange(action_start, char_start),
            c.input_action_mean,
            c.input_action_spread,
        )
        affect_edges = projection(
            "input_affect",
            np.arange(input_features),
            np.arange(affect_start, action_start),
            c.input_affect_mean,
            c.input_affect_spread,
        )
        affect_indices = affect_edges.reshape(
            input_features,
            len(self.layout.affect_subgroups),
            self.layout.subgroup_width(Population.AFFECT),
        ).transpose(1, 0, 2)
        affect_action_edges = synapses.weight.size
        projection(
            "affect_action",
            np.arange(affect_start, action_start),
            np.arange(action_start, char_start),
            c.affect_action_mean,
            c.affect_action_spread,
        )
        affect_action_indices = np.arange(affect_action_edges, synapses.weight.size)
        projection(
            "recurrent_output",
            np.arange(char_start, total),
            np.arange(char_start, total),
            c.recurrent_output_mean,
            c.recurrent_output_spread,
        )
        direct = projection(
            "direct_output",
            np.arange(input_features),
            np.arange(char_start, total),
            c.direct_output_mean,
            c.direct_output_spread,
        )
        hidden = projection(
            "hidden_output",
            np.arange(input_features, affect_start),
            np.arange(char_start, total),
            c.hidden_output_mean,
            c.hidden_output_spread,
        )
        self.network = NetworkCore(neurons, synapses)
        self.output_readout = EventReadout(self.layout)
        self.edge_enabled = np.ones(synapses.weight.size, dtype=bool)
        self.plasticity = RewardModulatedSTDP(synapses, learning_rate=config.learning_rate)
        self.response_session = ResponseSession(policy=config.session_policy)
        self.metrics = RuntimeMetrics(self.layout)
        self.homeostasis = PopulationHomeostasis(self.layout, config.homeostasis)
        self.external_drive = ArrayDrive(total)
        self.background_drive = BackgroundDrive(
            total,
            config.background_rate,
            config.background_current,
            initializer.seed_for("background"),
        )
        self.metrics_plugin = MetricsPlugin(self.metrics)
        self.plugins: list[NetworkPlugin] = [
            self.metrics_plugin,
            HomeostasisPlugin(self.homeostasis),
            PlasticityPlugin(self.plasticity),
        ]
        self._context = NetworkContext(self.network.tick, neurons.voltage, neurons.voltage)
        self.hidden_feature_groups = hidden_groups
        self.affect_edge_indices = affect_indices
        self.affect_action_edge_indices = affect_action_indices
        self.direct_input_output_edge_indices = direct
        self.hidden_output_edge_indices = hidden
        self.token_input_edge_indices = direct.reshape(
            input_features, len(output_tokens), self.layout.subgroup_width(Population.OUTPUT_CHAR)
        )
        self.hidden_recurrent_edge_indices = np.flatnonzero(
            (synapses.source >= input_features)
            & (synapses.source < affect_start)
            & (synapses.target >= input_features)
            & (synapses.target < affect_start)
        )
        self._ablation_lock = RLock()
        self.session = RuntimeSession(self)
        self.trainer = InputRunner(self)
        self.reset_diagnostics()

    def step(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.shape != (self.network.neurons.count,):
            raise ValueError(f"current must have shape ({self.network.neurons.count},)")
        self.external_drive.current[:] = current
        current_buffer = np.zeros_like(current)
        self.external_drive.add_to(current_buffer)
        self.background_drive.add_to(current_buffer)
        self.homeostasis.add_to(current_buffer)
        emitted = self.network.step(current_buffer)
        self._context.tick = self.network.tick
        self._context.spikes = emitted
        self._context.voltage = self.network.neurons.voltage
        for plugin in self.plugins:
            if self.network.tick % plugin.interval == 0:
                plugin.after_step(self._context)
        self.apply_ablation_mask()
        return emitted

    def reproducibility_snapshot(self) -> dict[str, object]:
        return self.session.reproducibility_snapshot()

    def restore_reproducibility_snapshot(self, snapshot: dict[str, object]) -> None:
        self.session.restore_reproducibility_snapshot(snapshot)

    @property
    def diagnostics(self) -> dict[str, float]:
        hidden = self.layout.slice(Population.HIDDEN)
        ticks = max(self.metrics.ticks, 1)
        return {
            "firing_rate": float(self.metrics.spikes[hidden].mean() / ticks),
            "output_firing_rate": float(
                self.metrics.spikes[self.layout.slice(Population.OUTPUT_CHAR)].mean() / ticks
            ),
            "output_event_rate": self.metrics.output_event_rate,
        }

    @property
    def population_diagnostics(self) -> dict[str, dict[str, float]]:
        result = {
            p.value: self.metrics.population(p, self.network.neurons.threshold).as_dict()
            for p in Population
        }
        result[Population.OUTPUT_CHAR.value]["output_event_rate"] = self.metrics.output_event_rate
        return result

    @property
    def pathway_diagnostics(self) -> dict[str, dict[str, float]]:
        pathways = {
            "direct_input_output": self.direct_input_output_edge_indices,
            "hidden_output": self.hidden_output_edge_indices,
            "hidden_recurrent": self.hidden_recurrent_edge_indices,
        }
        return {
            name: {
                "weight_l2": float(np.linalg.norm(self.network.synapses.weight[i])),
                "weight_mean_abs": float(np.mean(np.abs(self.network.synapses.weight[i])))
                if i.size
                else 0.0,
                "eligibility_l2": float(np.linalg.norm(self.plasticity.eligibility[i])),
                "eligibility_mean_abs": float(np.mean(np.abs(self.plasticity.eligibility[i])))
                if i.size
                else 0.0,
                "edge_count": float(i.size),
            }
            for name, i in pathways.items()
        }

    def reset_diagnostics(self) -> None:
        self.metrics.reset()

    def record_output_event(self, event: OutputEvent | None) -> None:
        if event is not None:
            self.metrics.record_output_event()

    def current(
        self, frame: np.ndarray, context: np.ndarray | None = None, *, tonic_affect: bool = False
    ) -> np.ndarray:
        current = np.zeros(self.network.neurons.count)
        current[: self.layout.input_count] = frame
        if context is not None:
            current[: self.layout.input_count] += context
        if tonic_affect and self.layout.affect_count:
            current[self.layout.slice(Population.AFFECT)] = 1.01
        return current

    def boundary_current(self, signal: InputSignal, strength: float = 5.0) -> np.ndarray:
        if signal not in (InputSignal.INPUT_BEGIN, InputSignal.INPUT_END):
            raise ValueError(f"unknown input signal: {signal}")
        current = np.zeros(self.layout.input_count)
        current[0 if signal is InputSignal.INPUT_BEGIN else 1] = strength
        return self.current(current)

    def step_input_signal(self, signal: InputSignal) -> np.ndarray:
        self.response_session.handle_input_signal(signal)
        return self.step(self.boundary_current(signal))

    def start_session(self, **state: object) -> None:
        self.session.start(**state)

    def finish_session(
        self,
        *,
        exhausted: bool,
        affect: object | None = None,
        working_memory: object | None = None,
    ) -> None:
        self.session.finish(exhausted=exhausted, affect=affect, working_memory=working_memory)

    def execute_isolated(
        self,
        operation: Callable[[SpikingRuntime], T],
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
        extension_hook: Callable[[SpikingRuntime, SpikingRuntime], None] | None = None,
    ) -> tuple[T, SessionExecutionSnapshot]:
        return self.session.execute_isolated(
            operation, conflict_policy=conflict_policy, merge=merge, extension_hook=extension_hook
        )

    def train_input_events(
        self,
        events: Iterable[InputSignal | np.ndarray | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.trainer.train_supervised(events, current_builder=current_builder)

    def train_delayed_copy_baseline(
        self,
        events: Iterable[InputSignal | np.ndarray | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Run the bounded supervised delayed-copy retention baseline."""
        self.trainer.train_supervised(
            events, current_builder=current_builder, hidden_retention=True
        )

    def train_reward_modulated_events(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        targets: Iterable[str] = (),
    ) -> tuple[OutputEvent, ...]:
        return self.trainer.train_reward_modulated(events, targets=targets)

    def run_input_events(
        self,
        events: Iterable[InputSignal | np.ndarray],
        *,
        response_ticks: int,
        observe_during_input: bool = True,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
        reward_callback: Callable[[OutputEvent], None] | None = None,
    ) -> tuple[OutputEvent, ...]:
        return self.trainer.run(
            events,
            response_ticks=response_ticks,
            observe_during_input=observe_during_input,
            current_builder=current_builder,
            reward_callback=reward_callback,
        )

    def _apply_supervised_target(self, frame: np.ndarray, target: str) -> None:
        self.trainer._supervised_target(frame, target)

    def _apply_boundary_target(self, target: str, emitted: np.ndarray) -> None:
        self.trainer._boundary_target(target, emitted)

    def ablate_edges(self, indices: np.ndarray) -> None:
        with self._ablation_lock:
            indices = np.asarray(indices, dtype=np.int64).ravel()
            if np.any(indices < 0) or np.any(indices >= self.edge_enabled.size):
                raise ValueError("ablation edge index is out of bounds")
            self.edge_enabled[indices] = False
            self.apply_ablation_mask()

    def apply_ablation_mask(self) -> None:
        with self._ablation_lock:
            self.network.synapses.weight[~self.edge_enabled] = 0.0
