"""Canonical composition root and efficient hot-path loop for the SNN."""

from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar

import numpy as np

from continual_agent.agent.input_runner import InputRunner
from continual_agent.agent.network_config import NetworkConfig
from continual_agent.agent.plugins import NetworkContext
from continual_agent.agent.runtime_session import RuntimeSession
from continual_agent.agent.session import (
    ConflictPolicy,
    InputSignal,
    SessionExecutionSnapshot,
)
from continual_agent.cognition.readout import OutputEvent
from continual_agent.simulation.population_layout import Population

T = TypeVar("T")


class SpikingRuntime:
    """Compose network services while keeping one vectorised tick loop."""

    layout: Any
    network: Any
    output_readout: Any
    edge_enabled: Any
    plasticity: Any
    response_session: Any
    metrics: Any
    metrics_plugin: Any
    homeostasis: Any
    external_drive: Any
    background_drive: Any
    drives: Any
    plugins: Any
    _context: NetworkContext
    input_features: int
    output_tokens: tuple[str, ...]
    neurons_per_token: int
    hidden_feature_groups: Any
    direct_input_output_edge_indices: Any
    hidden_output_edge_indices: Any
    hidden_recurrent_edge_indices: Any
    token_input_edge_indices: Any
    affect_edge_indices: Any
    affect_action_edge_indices: Any
    trainer: Any

    def __init__(self, config: NetworkConfig) -> None:
        bundle = config.build()
        self.__dict__.update(bundle.__dict__)
        self._context = bundle.context
        self.session = RuntimeSession(self)
        self.trainer = InputRunner(self)
        self.reset_diagnostics()

    def step(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.shape != (self.network.neurons.count,):
            raise ValueError(f"current must have shape ({self.network.neurons.count},)")
        self.external_drive.current[:] = current
        emitted = self.network.step(self.drives.collect())
        self._context.tick = self.network.tick
        self._context.spikes = emitted
        self._context.voltage = self.network.neurons.voltage
        for plugin in self.plugins:
            if self.network.tick % plugin.interval == 0:
                plugin.after_step(self._context)
        self.apply_ablation_mask()
        return emitted

    @property
    def network_state(self) -> dict[str, object]:
        return self.network.state_snapshot()

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
            self.metrics_plugin.record_output_event()

    def current(
        self, frame: np.ndarray, context: np.ndarray | None = None, *, tonic_affect: bool = False
    ) -> np.ndarray:
        current = np.zeros(self.network.neurons.count)
        current[: self.input_features] = frame
        if context is not None:
            current[: self.input_features] += context
        if tonic_affect and self.layout.affect_count:
            current[self.layout.slice(Population.AFFECT)] = 1.01
        return current

    def boundary_current(self, signal: InputSignal, strength: float = 5.0) -> np.ndarray:
        if signal not in (InputSignal.INPUT_BEGIN, InputSignal.INPUT_END):
            raise ValueError(f"unknown input signal: {signal}")
        current = np.zeros(self.input_features)
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

    def _snapshot(self, affect: object | None = None, working_memory: object | None = None):
        return self.session.snapshot(affect, working_memory)

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
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.trainer.train_supervised(events, current_builder=current_builder)

    def train_delayed_copy_baseline(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Run the bounded supervised delayed-copy retention baseline."""
        self.trainer.train_supervised(
            events, current_builder=current_builder, hidden_retention=True
        )

    def train_reward_modulated_events(
        self, events: Iterable[InputSignal | tuple[np.ndarray, str]]
    ) -> tuple[OutputEvent, ...]:
        return self.trainer.train_reward_modulated(events)

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
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if np.any(indices < 0) or np.any(indices >= self.edge_enabled.size):
            raise ValueError("ablation edge index is out of bounds")
        self.edge_enabled[indices] = False
        self.apply_ablation_mask()

    def apply_ablation_mask(self) -> None:
        self.network.synapses.weight[~self.edge_enabled] = 0.0
