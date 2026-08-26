"""Task-neutral spiking network runtime shared by production experiments."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import Callable, Iterable, TypeVar, cast

import numpy as np

from continual_agent.agent.session import (
    ConflictPolicy,
    InputSignal,
    ResponseSession,
    SessionExecutionSnapshot,
    SessionPolicy,
    SessionSnapshot,
    SessionState,
)
from continual_agent.cognition.readout import EventReadout, OutputEvent
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.network import SpikingNetwork
from continual_agent.simulation.neurons import LIFNeurons
from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses

T = TypeVar("T")


class SpikingRuntime:
    """Own network mechanics while callers provide task-specific semantics."""

    def __init__(
        self,
        *,
        input_features: int,
        hidden_neurons: int,
        output_tokens: tuple[str, ...],
        neurons_per_token: int,
        action_names: tuple[str, ...] = (),
        neurons_per_action: int = 1,
        affect_names: tuple[str, ...] = (),
        neurons_per_affect: int = 1,
        connection_probability: float = 0.08,
        seed: int = 0,
        learning_rate: float = 0.08,
        session_policy: SessionPolicy | None = None,
    ) -> None:
        self.input_features = input_features
        self.output_tokens = output_tokens
        self.neurons_per_token = neurons_per_token
        affect_start = input_features + hidden_neurons
        action_start = affect_start + len(affect_names) * neurons_per_affect
        char_start = action_start + len(action_names) * neurons_per_action
        total = char_start + len(output_tokens) * neurons_per_token
        self.layout = PopulationLayout(
            input_count=input_features,
            hidden_count=hidden_neurons,
            affect_count=action_start - affect_start,
            action_count=char_start - action_start,
            char_count=total - char_start,
            affect_subgroups={
                name: slice(
                    affect_start + i * neurons_per_affect,
                    affect_start + (i + 1) * neurons_per_affect,
                )
                for i, name in enumerate(affect_names)
            },
            action_subgroups={
                name: slice(
                    action_start + i * neurons_per_action,
                    action_start + (i + 1) * neurons_per_action,
                )
                for i, name in enumerate(action_names)
            },
            char_subgroups={
                name: slice(
                    char_start + i * neurons_per_token, char_start + (i + 1) * neurons_per_token
                )
                for i, name in enumerate(output_tokens)
            },
        )
        self.neurons = LIFNeurons(
            total, dt=1.0, tau_membrane=5.0, threshold=1.0, refractory_ticks=2
        )
        rng = np.random.default_rng(seed)
        self.synapses = SparseSynapses.random(
            total,
            connection_probability,
            rng,
            excitatory_weight=0.18,
            inhibitory_weight=-0.12,
            inhibitory_fraction=0.15,
        )
        input_hidden_edges = self.synapses.weight.size
        self.hidden_feature_groups = tuple(
            np.asarray(group, dtype=np.int64)
            for group in np.array_split(np.arange(input_features, affect_start), input_features)
        )
        for source, group in enumerate(self.hidden_feature_groups):
            self._add_projection(rng, np.array([source]), group, 3.0, 0.0)
        self.input_hidden_edge_indices = np.arange(input_hidden_edges, self.synapses.weight.size)
        self._add_projection(
            rng, np.arange(input_features), np.arange(action_start, char_start), 0.9, 0.08
        )
        affect_edges = self.synapses.weight.size
        self._add_projection(
            rng, np.arange(input_features), np.arange(affect_start, action_start), 0.0, 0.02
        )
        affect_block = np.arange(affect_edges, self.synapses.weight.size)
        self.affect_edge_indices = affect_block.reshape(
            input_features, len(affect_names), neurons_per_affect
        ).transpose(1, 0, 2)
        affect_action_edges = self.synapses.weight.size
        self._add_projection(
            rng,
            np.arange(affect_start, action_start),
            np.arange(action_start, char_start),
            0.0,
            0.02,
        )
        self.affect_action_edge_indices = np.arange(affect_action_edges, self.synapses.weight.size)
        token_edges = self.synapses.weight.size
        self._add_projection(rng, np.arange(char_start), np.arange(char_start, total), 0.0, 0.025)
        self.direct_input_output_edge_indices = (
            token_edges
            + self._projection_indices(
                input_features, len(output_tokens), neurons_per_token
            ).ravel()
        )
        hidden_offset = hidden_start = input_features
        self.hidden_output_edge_indices = (
            token_edges
            + self._projection_indices(
                hidden_offset + hidden_neurons, len(output_tokens), neurons_per_token
            )[hidden_start : hidden_start + hidden_neurons].ravel()
        )
        self.token_input_edge_indices = self.direct_input_output_edge_indices.reshape(
            input_features, len(output_tokens), neurons_per_token
        )
        self.recurrent_event_edge_indices = np.flatnonzero(
            (self.synapses.source >= input_features) & (self.synapses.target >= char_start)
        )
        self.network = SpikingNetwork(self.neurons, self.synapses)
        self.output_readout = EventReadout(self.layout)
        self.edge_enabled = np.ones(self.synapses.weight.size, dtype=bool)
        self.plasticity = RewardModulatedSTDP(self.synapses, learning_rate=learning_rate)
        self.response_session = ResponseSession(policy=session_policy or SessionPolicy())

    def _add_projection(
        self,
        rng: np.random.Generator,
        sources: np.ndarray,
        targets: np.ndarray,
        mean: float,
        spread: float,
    ) -> None:
        source = np.repeat(sources, targets.size)
        target = np.tile(targets, sources.size)
        self.synapses.add_edges(source, target, rng.normal(mean, spread, source.size))

    @staticmethod
    def _projection_indices(
        source_count: int, token_count: int, neurons_per_token: int
    ) -> np.ndarray:
        size = token_count * neurons_per_token
        return np.array(
            [
                np.arange(i * size, (i + 1) * size).reshape(token_count, neurons_per_token)
                for i in range(source_count)
            ]
        )

    @property
    def network_state(self) -> dict[str, object]:
        return cast(dict[str, object], self.network.state_snapshot())

    def step(self, current: np.ndarray) -> np.ndarray:
        emitted = self.network.step(current)
        self.plasticity.observe(emitted)
        self.apply_ablation_mask()
        return emitted

    def ablate_edges(self, indices: np.ndarray) -> None:
        """Disable edges persistently, including during subsequent learning."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if np.any(indices < 0) or np.any(indices >= self.edge_enabled.size):
            raise ValueError("ablation edge index is out of bounds")
        self.edge_enabled[indices] = False
        self.apply_ablation_mask()

    def apply_ablation_mask(self) -> None:
        self.synapses.weight[~self.edge_enabled] = 0.0

    def current(
        self, frame: np.ndarray, context: np.ndarray | None = None, *, tonic_affect: bool = False
    ) -> np.ndarray:
        current = np.zeros(self.neurons.count)
        current[: self.input_features] = frame
        if context is not None:
            current[: self.input_features] += context
        if tonic_affect and self.layout.affect_count:
            current[self.layout.slice(Population.AFFECT)] = 1.01
        return current

    def _snapshot(
        self, affect: object | None = None, working_memory: object | None = None
    ) -> SessionSnapshot:
        state = self.network.state_snapshot()
        voltage = state["voltage"]
        refractory = state["refractory"]
        pending = state["pending_current"]
        assert isinstance(voltage, np.ndarray)
        assert isinstance(refractory, np.ndarray)
        assert isinstance(pending, np.ndarray)
        return SessionSnapshot(
            voltage,
            refractory,
            pending,
            affect=affect,
            working_memory=working_memory,
            plasticity_eligibility=self.plasticity.eligibility,
        )

    def start_session(
        self, *, affect: object | None = None, working_memory: object | None = None
    ) -> None:
        if self.response_session.state in (
            SessionState.COMPLETE,
            SessionState.EOS_RECEIVED,
            SessionState.EXHAUSTED,
        ):
            self.response_session.reset_for_next_response()
        elif self.response_session.state is SessionState.RESPONDING:
            self.response_session.abort()
        policy = self.response_session.policy
        if policy.reset_neuron_state and policy.reset_synaptic_activity:
            self.network.reset_state()
        elif policy.reset_neuron_state:
            self.network.reset_neuron_state()
        elif policy.reset_synaptic_activity:
            self.network.reset_synaptic_activity()
        if not policy.persist_plasticity_eligibility:
            self.plasticity.reset_traces()
        if policy.reset_readout:
            self.output_readout.reset()
        self.response_session.begin(self._snapshot(affect, working_memory))

    def finish_session(
        self, *, exhausted: bool, affect: object | None = None, working_memory: object | None = None
    ) -> None:
        snapshot = self._snapshot(affect, working_memory)
        if exhausted:
            self.response_session.exhaust(snapshot)
        else:
            self.response_session.receive_eos()
            self.response_session.complete(snapshot)

    def train_input_events(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
        *,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Train one explicitly delimited input presentation.

        A failed or malformed presentation always leaves the runtime ready for
        the next one; an unfinished session is never reused accidentally.
        """
        if self.response_session.state is not SessionState.IDLE:
            self.response_session = ResponseSession(policy=self.response_session.policy)
        self.start_session()
        build_current = current_builder or self.current
        saw_begin = saw_end = False
        try:
            for item in events:
                if isinstance(item, InputSignal):
                    if item is InputSignal.INPUT_BEGIN and saw_begin:
                        raise ValueError("training input contains multiple INPUT_BEGIN signals")
                    if item is InputSignal.INPUT_END and not saw_begin:
                        raise ValueError("training input ended before INPUT_BEGIN")
                    self.response_session.handle_input_signal(item)
                    saw_begin |= item is InputSignal.INPUT_BEGIN
                    saw_end |= item is InputSignal.INPUT_END
                    continue
                if not saw_begin or saw_end:
                    raise ValueError("training frames must be between INPUT_BEGIN and INPUT_END")
                self.response_session.accept_input_frame()
                frame, target = item
                frame = np.asarray(frame, dtype=float)
                if frame.shape != (self.input_features,):
                    raise ValueError("raw input frame has the wrong shape")
                if not isinstance(target, str) or target not in self.output_tokens:
                    raise ValueError("labelled raw input frames require a known string target")
                self.step(build_current(frame))
                active = 2.0 * np.maximum(frame, 0.0)
                target_index = self.output_tokens.index(target)
                hidden_activity = np.maximum(
                    self.network.neurons.voltage[self.layout.slice(Population.HIDDEN)], 0.0
                )
                active_feature = int(np.argmax(active))
                group = self.hidden_feature_groups[active_feature] - self.input_features
                hidden_activity[group] += active.mean()
                edges = self.hidden_output_edge_indices.reshape(
                    hidden_activity.size,
                    len(self.output_tokens),
                    self.neurons_per_token,
                )[:, target_index]
                selected_edges = edges[group]
                self.synapses.weight[selected_edges] = np.clip(
                    self.synapses.weight[selected_edges]
                    + self.plasticity.learning_rate
                    * hidden_activity[group, None]
                    / selected_edges.shape[-1],
                    -1.0,
                    1.0,
                )
                self.apply_ablation_mask()
                teacher = build_current(np.zeros(self.input_features))
                teacher[self.layout.subgroup(Population.OUTPUT_CHAR, target)] += 3.0
                self.step(teacher)
            if not saw_begin or not saw_end or self.response_session.input_active:
                raise ValueError("training input must contain INPUT_BEGIN and INPUT_END boundaries")
        finally:
            self.plasticity.reset_traces()
            self.network.reset_state()
            self.output_readout.reset()
            self.apply_ablation_mask()
            self.response_session = ResponseSession(policy=self.response_session.policy)

    def train_reward_modulated_events(
        self,
        events: Iterable[InputSignal | tuple[np.ndarray, str]],
    ) -> tuple[OutputEvent, ...]:
        """Present labelled targets as a teaching signal, then leave eligibility intact.

        No synapse is changed here.  The caller supplies the delayed scalar reward
        to ``plasticity.reinforce`` after evaluating the resulting episode.
        """
        if self.response_session.state is not SessionState.IDLE:
            self.response_session = ResponseSession(policy=self.response_session.policy)
        self.start_session()
        build_current = self.current
        observed: list[OutputEvent] = []
        saw_begin = saw_end = False
        try:
            for item in events:
                if isinstance(item, InputSignal):
                    self.response_session.handle_input_signal(item)
                    saw_begin |= item is InputSignal.INPUT_BEGIN
                    saw_end |= item is InputSignal.INPUT_END
                    continue
                if not saw_begin or saw_end:
                    raise ValueError("training frames must be between input boundaries")
                self.response_session.accept_input_frame()
                frame, target = item
                frame = np.asarray(frame, dtype=float)
                self.step(build_current(frame))
                teacher = build_current(np.zeros(self.input_features))
                teacher[self.layout.subgroup(Population.OUTPUT_CHAR, target)] += 3.0
                emitted = self.step(teacher)
                event = self.output_readout.observe(
                    emitted, activation=self.neurons.voltage, populations=(Population.OUTPUT_CHAR,)
                )
                if event is not None:
                    observed.append(event)
            if not saw_begin or not saw_end:
                raise ValueError("training input must contain input boundaries")
            self.response_session.begin_response()
            self.response_session.abort(self._snapshot())
            return tuple(observed)
        except Exception:
            self.plasticity.reset_traces()
            self.network.reset_state()
            self.output_readout.reset()
            self.response_session = ResponseSession(policy=self.response_session.policy)
            raise

    def run_input_events(
        self,
        events: Iterable[InputSignal | np.ndarray],
        *,
        response_ticks: int,
        observe_during_input: bool = True,
        current_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[OutputEvent, ...]:
        if response_ticks <= 0:
            raise ValueError("response_ticks must be positive")
        if self.response_session.state is not SessionState.IDLE:
            self.response_session = ResponseSession(policy=self.response_session.policy)
        self.start_session()
        build_current = current_builder or self.current
        completed = False
        try:
            for item in events:
                if isinstance(item, InputSignal):
                    self.response_session.handle_input_signal(item)
                    continue
                frame = np.asarray(item, dtype=float)
                if frame.shape != (self.input_features,):
                    raise ValueError("raw input frame has the wrong shape")
                self.response_session.accept_input_frame()
                emitted = self.step(build_current(frame))
                if observe_during_input:
                    self.output_readout.observe(
                        emitted,
                        activation=self.neurons.voltage,
                        populations=(Population.OUTPUT_CHAR,),
                    )
            self.response_session.begin_response()
            for _ in range(response_ticks):
                emitted = self.step(build_current(np.zeros(self.input_features)))
                self.output_readout.observe(
                    emitted, activation=self.neurons.voltage, populations=(Population.OUTPUT_CHAR,)
                )
            self.response_session.abort(self._snapshot())
            completed = True
            return tuple(self.output_readout.events)
        finally:
            if not completed:
                # A malformed stream must not strand lifecycle or transient neural state.
                self.plasticity.reset_traces()
                self.network.reset_state()
                self.output_readout.reset()
                self.apply_ablation_mask()
                self.response_session = ResponseSession(policy=self.response_session.policy)

    def execute_isolated(
        self,
        operation: Callable[["SpikingRuntime"], T],
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
        extension_hook: Callable[["SpikingRuntime", "SpikingRuntime"], None] | None = None,
    ) -> tuple[T, SessionExecutionSnapshot]:
        baseline = self._snapshot()
        execution = SessionExecutionSnapshot(baseline, self.synapses.weight)
        isolated = self.clone_state(extension_hook)
        result = operation(isolated)
        changed = np.flatnonzero(
            ~np.isclose(isolated.synapses.weight, execution.weights, rtol=0.0, atol=0.0)
        )
        if changed.size:
            execution.record_weight_update(changed, isolated.synapses.weight[changed])
        if merge:
            execution.merge_into(self.synapses.weight, conflict_policy)
        return result, execution

    def clone_state(
        self, extension_hook: Callable[["SpikingRuntime", "SpikingRuntime"], None] | None = None
    ) -> "SpikingRuntime":
        """Clone runtime-owned state, then let a task copy its own state."""
        isolated = copy(self)
        isolated.layout = self.layout
        isolated.neurons = deepcopy(self.neurons)
        isolated.network = self.network.copy()
        isolated.neurons = isolated.network.neurons
        isolated.synapses = isolated.network.synapses
        isolated.plasticity = deepcopy(self.plasticity)
        isolated.plasticity.synapses = isolated.synapses
        isolated.output_readout = EventReadout(
            self.layout,
            activation_threshold=self.output_readout.activation_threshold,
            release_threshold=self.output_readout.release_threshold,
            arbitration=deepcopy(self.output_readout.arbitration),
        )
        isolated.response_session = deepcopy(self.response_session)
        isolated.network.restore_state(self.network.state_snapshot())
        isolated.synapses.weight[:] = self.synapses.weight
        isolated.edge_enabled = self.edge_enabled.copy()
        isolated.apply_ablation_mask()
        if extension_hook is not None:
            extension_hook(self, isolated)
        return isolated
