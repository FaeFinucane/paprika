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
        events: Iterable[InputSignal | np.ndarray | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
        hidden_retention: bool = False,
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
                if isinstance(item, tuple):
                    frame, target = item
                else:
                    frame, target = item, None
                frame = np.asarray(frame, dtype=float)
                if frame.shape != (runtime.layout.input_count,):
                    raise ValueError("raw input frame has the wrong shape")
                if target is not None and target not in runtime.layout.char_subgroups:
                    raise ValueError("labelled raw input frames require a known string target")
                runtime.step(build(frame))
                if target is not None:
                    runtime._apply_supervised_target(frame, target)
                if hidden_retention and target is not None:
                    self._apply_delayed_copy_baseline(frame, target)
                # Apply the teacher as a weight target, without inventing a tick.
            if ended:
                emitted = runtime.step(build(np.zeros(runtime.layout.input_count)))
                runtime._apply_boundary_target("<EOS>", emitted)
                teacher = build(np.zeros(runtime.layout.input_count))
                teacher[runtime.layout.subgroup(Population.OUTPUT_CHAR, "<EOS>")] += 3.0
                runtime.step(teacher)
            if not began or not ended or runtime.response_session.input_active:
                raise ValueError("training input must contain INPUT_BEGIN and INPUT_END boundaries")
        finally:
            self._reset()

    def _apply_delayed_copy_baseline(self, frame: np.ndarray, target: str) -> None:
        """Teach the active hidden group to retain and emit its symbol."""
        runtime = self.runtime
        feature = int(np.argmax(np.maximum(frame, 0.0)))
        group = runtime.hidden_feature_groups[feature] - runtime.layout.input_count
        edges = runtime.hidden_recurrent_edge_indices
        sources = runtime.network.synapses.source[edges] - runtime.layout.input_count
        targets = runtime.network.synapses.target[edges] - runtime.layout.input_count
        selected = edges[np.isin(sources, group) & np.isin(targets, group)]
        if selected.size == 0:
            return
        # Use the presented feature as the bounded structural teaching signal;
        # recurrence is retained even when the sparse hidden group is quiet.
        signal = float(np.clip(np.maximum(frame, 0.0).mean(), 0.0, 1.0))
        update = np.full(selected.size, runtime.plasticity.learning_rate * signal)
        runtime.network.synapses.weight[selected] = np.clip(
            runtime.network.synapses.weight[selected] + update, -1.0, 1.0
        )
        output_edges = runtime.hidden_output_edge_indices.reshape(
            runtime.layout.hidden_count,
            len(runtime.layout.char_subgroups),
            runtime.layout.subgroup_width(Population.OUTPUT_CHAR),
        )
        target_edges = output_edges[group, tuple(runtime.layout.char_subgroups).index(target)]
        runtime.network.synapses.weight[target_edges] = 1.0
        runtime.network.neurons.voltage[runtime.layout.slice(Population.HIDDEN)][group] = 1.0
        runtime.apply_ablation_mask()

    def train_reward_modulated(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        targets: Iterable[str] = (),
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
                    frame.shape != (runtime.layout.input_count,)
                    or not isinstance(target, str)
                    or target not in runtime.layout.char_subgroups
                ):
                    raise ValueError("labelled raw input frames require a known string target")
                emitted = runtime.step(runtime.current(frame))
                event = runtime.output_readout.observe(
                    emitted,
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
            target_values = tuple(targets)
            if target_values:
                from continual_agent.evaluation.event_stream import evaluate_event_stream

                report = evaluate_event_stream(target_values, tuple(observed), end_time=clock)
                runtime.plasticity.reinforce(report.total_reward)
            runtime.plasticity.reset_traces()
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
                        visible = emitted
                        if item is InputSignal.INPUT_END and not observe_during_input:
                            eos = runtime.layout.subgroup(Population.OUTPUT_CHAR, "<EOS>")
                            visible = np.zeros_like(emitted, dtype=float)
                            visible[eos] = emitted[eos]
                        event = runtime.output_readout.observe(
                            visible,
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
                if frame.shape != (runtime.layout.input_count,):
                    raise ValueError("raw input frame has the wrong shape")
                runtime.response_session.accept_input_frame()
                emitted = runtime.step(build(frame))
                if observe_during_input:
                    event = runtime.output_readout.observe(
                        emitted,
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
                emitted = runtime.step(build(np.zeros(runtime.layout.input_count)))
                event = runtime.output_readout.observe(
                    emitted,
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
        if policy.reset_membrane or policy.reset_refractory:
            runtime.network.reset_neuron_state()
        if policy.reset_pending_current or policy.reset_recurrent_activity:
            runtime.network.reset_synaptic_activity()
        if policy.reset_eligibility:
            runtime.plasticity.reset_traces()
        if policy.reset_background_rng:
            runtime.background_drive.reset_rng()
        if (
            policy.reset_membrane
            or policy.reset_refractory
            or policy.reset_pending_current
            or policy.reset_recurrent_activity
        ):
            runtime.network.tick = 0
        runtime.output_readout.reset()
        runtime.apply_ablation_mask()
        runtime.response_session = ResponseSession(policy=runtime.response_session.policy)

    def _supervised_target(self, frame: np.ndarray, target: str) -> None:
        runtime = self.runtime
        active = 2.0 * np.maximum(frame, 0.0)
        output_tokens = tuple(runtime.layout.char_subgroups)
        target_index = output_tokens.index(target)
        hidden_activity = np.maximum(
            runtime.network.neurons.voltage[runtime.layout.slice(Population.HIDDEN)], 0.0
        )
        feature = int(np.argmax(active))
        group = runtime.hidden_feature_groups[feature] - runtime.layout.input_count
        hidden_activity[group] += active.mean()
        edges = runtime.hidden_output_edge_indices.reshape(
            hidden_activity.size,
            len(output_tokens),
            runtime.layout.subgroup_width(Population.OUTPUT_CHAR),
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
        output_tokens = tuple(runtime.layout.char_subgroups)
        hidden = runtime.layout.slice(Population.HIDDEN)
        activity = np.maximum(runtime.network.neurons.voltage[hidden], 0.0) + emitted[
            hidden
        ].astype(float)
        if not np.any(activity):
            return
        edges = runtime.hidden_output_edge_indices.reshape(
            activity.size, len(output_tokens), runtime.layout.subgroup_width(Population.OUTPUT_CHAR)
        )[:, output_tokens.index(target)]
        selected = edges[activity > 0.0]
        runtime.network.synapses.weight[selected] = np.clip(
            runtime.network.synapses.weight[selected]
            + runtime.plasticity.learning_rate
            * activity[activity > 0.0, None]
            / runtime.layout.subgroup_width(Population.OUTPUT_CHAR),
            -1.0,
            1.0,
        )
        runtime.apply_ablation_mask()
