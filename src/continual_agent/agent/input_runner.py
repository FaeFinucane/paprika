"""Raw input episode execution and network-level training protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from continual_agent.agent.session import InputSignal, ResponseSession, SessionState
from continual_agent.cognition.readout import OutputEvent
from continual_agent.simulation.population_layout import Population

if TYPE_CHECKING:
    from continual_agent.agent.spiking_runtime import SpikingRuntime


class InputRunner:
    def __init__(self, runtime: SpikingRuntime) -> None:
        self.runtime = runtime

    def train_supervised(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        runtime = self.runtime
        self._prepare()
        build = current_builder or runtime.current
        began = ended = False
        try:
            for item in events:
                if isinstance(item, InputSignal):
                    if item is InputSignal.INPUT_BEGIN and began:
                        raise ValueError("training input contains multiple INPUT_BEGIN signals")
                    if item is InputSignal.INPUT_END and not began:
                        raise ValueError("training input ended before INPUT_BEGIN")
                    runtime.step_input_signal(item)
                    began |= item is InputSignal.INPUT_BEGIN
                    ended |= item is InputSignal.INPUT_END
                    continue
                if not began or ended:
                    raise ValueError("training frames must be between INPUT_BEGIN and INPUT_END")
                runtime.response_session.accept_input_frame()
                frame, target = item
                frame = np.asarray(frame, dtype=float)
                if frame.shape != (runtime.input_features,):
                    raise ValueError("raw input frame has the wrong shape")
                if not isinstance(target, str) or target not in runtime.output_tokens:
                    raise ValueError("labelled raw input frames require a known string target")
                runtime.step(build(frame))
                runtime._apply_supervised_target(frame, target)
                teacher = build(np.zeros(runtime.input_features))
                teacher[runtime.layout.subgroup(Population.OUTPUT_CHAR, target)] += 3.0
                runtime.step(teacher)
            if ended:
                emitted = runtime.step(build(np.zeros(runtime.input_features)))
                runtime._apply_boundary_target("<EOS>", emitted)
                teacher = build(np.zeros(runtime.input_features))
                teacher[runtime.layout.subgroup(Population.OUTPUT_CHAR, "<EOS>")] += 3.0
                runtime.step(teacher)
            if not began or not ended or runtime.response_session.input_active:
                raise ValueError("training input must contain INPUT_BEGIN and INPUT_END boundaries")
        finally:
            self._reset()

    def train_reward_modulated(
        self, events: Iterable[InputSignal | tuple[np.ndarray, str]]
    ) -> tuple[OutputEvent, ...]:
        runtime = self.runtime
        self._prepare()
        observed: list[OutputEvent] = []
        began = ended = False
        clock = 0
        try:
            for item in events:
                if isinstance(item, InputSignal):
                    emitted = runtime.step_input_signal(item)
                    event = runtime.output_readout.observe(
                        emitted,
                        activation=runtime.network.neurons.voltage,
                        populations=(Population.OUTPUT_CHAR,),
                        timestamp=clock,
                    )
                    runtime.record_output_event(event)
                    if event is not None:
                        observed.append(event)
                    clock += 1
                    began |= item is InputSignal.INPUT_BEGIN
                    ended |= item is InputSignal.INPUT_END
                    continue
                if not began or ended:
                    raise ValueError("training frames must be between input boundaries")
                runtime.response_session.accept_input_frame()
                frame, target = item
                frame = np.asarray(frame, dtype=float)
                if (
                    frame.shape != (runtime.input_features,)
                    or not isinstance(target, str)
                    or target not in runtime.output_tokens
                ):
                    raise ValueError("labelled raw input frames require a known string target")
                emitted = runtime.step(runtime.current(frame))
                event = runtime.output_readout.observe(
                    emitted,
                    activation=runtime.network.neurons.voltage,
                    populations=(Population.OUTPUT_CHAR,),
                    timestamp=clock,
                )
                runtime.record_output_event(event)
                if event is not None:
                    observed.append(event)
                clock += 1
            if not began or not ended:
                raise ValueError("training input must contain input boundaries")
            runtime.response_session.begin_response()
            runtime.response_session.abort(runtime.session.snapshot())
            return tuple(observed)
        except Exception:
            self._reset()
            raise

    def run(
        self,
        events: Iterable[InputSignal | np.ndarray],
        *,
        response_ticks: int,
        observe_during_input: bool = True,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
        reward_callback: Callable[[OutputEvent], None] | None = None,
    ) -> tuple[OutputEvent, ...]:
        if response_ticks <= 0:
            raise ValueError("response_ticks must be positive")
        runtime = self.runtime
        self._prepare()
        build = current_builder or runtime.current
        complete = False
        try:
            clock = 0
            for item in events:
                if isinstance(item, InputSignal):
                    emitted = runtime.step_input_signal(item)
                    if item is InputSignal.INPUT_END or observe_during_input:
                        visible, activation = emitted, runtime.network.neurons.voltage
                        if item is InputSignal.INPUT_END and not observe_during_input:
                            eos = runtime.layout.subgroup(Population.OUTPUT_CHAR, "<EOS>")
                            visible = np.zeros_like(emitted, dtype=float)
                            visible[eos] = emitted[eos]
                            activation = np.zeros_like(runtime.network.neurons.voltage)
                            activation[eos] = runtime.network.neurons.voltage[eos]
                        event = runtime.output_readout.observe(
                            visible,
                            activation=activation,
                            populations=(Population.OUTPUT_CHAR,),
                            timestamp=clock,
                        )
                        runtime.record_output_event(event)
                        if event is not None and reward_callback is not None:
                            reward_callback(event)
                    if item is InputSignal.INPUT_END:
                        runtime.response_session.begin_response()
                    clock += 1
                    continue
                frame = np.asarray(item, dtype=float)
                if frame.shape != (runtime.input_features,):
                    raise ValueError("raw input frame has the wrong shape")
                runtime.response_session.accept_input_frame()
                emitted = runtime.step(build(frame))
                if observe_during_input:
                    event = runtime.output_readout.observe(
                        emitted,
                        activation=runtime.network.neurons.voltage,
                        populations=(Population.OUTPUT_CHAR,),
                        timestamp=clock,
                    )
                    runtime.record_output_event(event)
                    if event is not None and reward_callback is not None:
                        reward_callback(event)
                clock += 1
            if runtime.response_session.state is SessionState.RECEIVING_INPUT:
                runtime.response_session.begin_response()
            for _ in range(response_ticks):
                emitted = runtime.step(build(np.zeros(runtime.input_features)))
                event = runtime.output_readout.observe(
                    emitted,
                    activation=runtime.network.neurons.voltage,
                    populations=(Population.OUTPUT_CHAR,),
                    timestamp=clock,
                )
                runtime.record_output_event(event)
                if event is not None and reward_callback is not None:
                    reward_callback(event)
                clock += 1
            runtime.response_session.abort(runtime.session.snapshot())
            complete = True
            return tuple(runtime.output_readout.events)
        finally:
            if not complete:
                self._reset()

    def _prepare(self) -> None:
        runtime = self.runtime
        if runtime.response_session.state is not SessionState.IDLE:
            runtime.response_session = ResponseSession(policy=runtime.response_session.policy)
        runtime.session.start()

    def _reset(self) -> None:
        runtime = self.runtime
        policy = runtime.response_session.policy
        if policy.reset_neuron_state and policy.reset_synaptic_activity:
            runtime.network.reset_state()
        elif policy.reset_neuron_state:
            runtime.network.reset_neuron_state()
        elif policy.reset_synaptic_activity:
            runtime.network.reset_synaptic_activity()
        if not policy.persist_plasticity_eligibility:
            runtime.plasticity.reset_traces()
        runtime.output_readout.reset()
        runtime.apply_ablation_mask()
        runtime.response_session = ResponseSession(policy=runtime.response_session.policy)

    def _supervised_target(self, frame: np.ndarray, target: str) -> None:
        runtime = self.runtime
        active = 2.0 * np.maximum(frame, 0.0)
        target_index = runtime.output_tokens.index(target)
        hidden_activity = np.maximum(
            runtime.network.neurons.voltage[runtime.layout.slice(Population.HIDDEN)], 0.0
        )
        feature = int(np.argmax(active))
        group = runtime.hidden_feature_groups[feature] - runtime.input_features
        hidden_activity[group] += active.mean()
        edges = runtime.hidden_output_edge_indices.reshape(
            hidden_activity.size, len(runtime.output_tokens), runtime.neurons_per_token
        )[:, target_index][group]
        runtime.network.synapses.weight[edges] = np.clip(
            runtime.network.synapses.weight[edges]
            + runtime.plasticity.learning_rate * hidden_activity[group, None] / edges.shape[-1],
            -1.0,
            1.0,
        )
        runtime.apply_ablation_mask()

    def _boundary_target(self, target: str, emitted: np.ndarray) -> None:
        runtime = self.runtime
        hidden = runtime.layout.slice(Population.HIDDEN)
        activity = np.maximum(runtime.network.neurons.voltage[hidden], 0.0) + emitted[
            hidden
        ].astype(float)
        if not np.any(activity):
            return
        edges = runtime.hidden_output_edge_indices.reshape(
            activity.size, len(runtime.output_tokens), runtime.neurons_per_token
        )[:, runtime.output_tokens.index(target)]
        selected = edges[activity > 0.0]
        runtime.network.synapses.weight[selected] = np.clip(
            runtime.network.synapses.weight[selected]
            + runtime.plasticity.learning_rate
            * activity[activity > 0.0, None]
            / runtime.neurons_per_token,
            -1.0,
            1.0,
        )
        runtime.apply_ablation_mask()
