"""A text-only vertical slice around the spiking network."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

import numpy as np

from continual_agent.agent.debug import DebugSnapshot
from continual_agent.agent.session import (
    ConflictPolicy,
    ResponseSession,
    SessionPolicy,
    SessionExecutionSnapshot,
    SessionSnapshot,
    SessionState,
)
from continual_agent.cognition.affect import AffectiveEvent, AffectiveState
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.cognition.readout import Action, ActionReadout, Decision, EventReadout
from continual_agent.cognition.working_memory import WorkingMemory
from continual_agent.encoding.text_encoder import TextEncoder
from continual_agent.environment.protocol import AgentAction
from continual_agent.language.spiking_decoder import GeneratedResponse, SpikingCharacterDecoder
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.neurons import LIFNeurons
from continual_agent.simulation.network import SpikingNetwork
from continual_agent.simulation.synapses import SparseSynapses
from continual_agent.simulation.population_layout import Population, PopulationLayout

if TYPE_CHECKING:
    from continual_agent.evaluation.event_stream import (
        AccountedEvent,
        EventStreamConfig,
        EventStreamReport,
        SilenceInterval,
        TargetEvent,
    )


T = TypeVar("T")


@dataclass(frozen=True)
class AgentConfig:
    input_features: int = 48
    neurons_per_action: int = 4
    neurons_per_affect: int = 4
    hidden_neurons: int = 48
    max_thinking_ticks: int = 18
    connection_probability: float = 0.08
    seed: int = 0
    persistent_working_memory: bool = True
    neurons_per_token: int = 3
    max_response_tokens: int = 48
    max_response_ticks: int = 144
    # Values above one hold each encoded frame longer for early teaching.
    presentation_speed: int = 1
    language_alphabet: tuple[str, ...] = (
        "m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!"
    )


class ConversationAgent:
    """Continuously plastic text agent with a small typed action space."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.actions = tuple(Action)
        self.language = SpikingCharacterDecoder(
            alphabet=self.config.language_alphabet,
            neurons_per_token=self.config.neurons_per_token,
        )
        affect_start = self.config.input_features + self.config.hidden_neurons
        action_start = affect_start + len(AffectiveCircuit.signal_names) * self.config.neurons_per_affect
        token_start = action_start + len(self.actions) * self.config.neurons_per_action
        token_end = token_start + self.language.neuron_count
        self.layout = PopulationLayout(
            input_count=self.config.input_features,
            hidden_count=self.config.hidden_neurons,
            affect_count=action_start - affect_start,
            action_count=token_start - action_start,
            char_count=token_end - token_start,
            affect_subgroups={name: slice(affect_start + i * self.config.neurons_per_affect, affect_start + (i + 1) * self.config.neurons_per_affect) for i, name in enumerate(AffectiveCircuit.signal_names)},
            action_subgroups={name.value: slice(action_start + i * self.config.neurons_per_action, action_start + (i + 1) * self.config.neurons_per_action) for i, name in enumerate(self.actions)},
            char_subgroups={token: slice(token_start + i * self.config.neurons_per_token, token_start + (i + 1) * self.config.neurons_per_token) for i, token in enumerate(self.language.tokens)},
        )
        affect_count = self.layout.affect_count
        affect = self.layout.slice(Population.AFFECT)
        action = self.layout.slice(Population.OUTPUT_ACTION)
        chars = self.layout.slice(Population.OUTPUT_CHAR)
        neuron_count = self.layout.total_count
        rng = np.random.default_rng(self.config.seed)
        neurons = LIFNeurons(
            count=neuron_count,
            dt=1.0,
            tau_membrane=5.0,
            threshold=1.0,
            refractory_ticks=2,
        )
        synapses = SparseSynapses.random(
            neuron_count,
            self.config.connection_probability,
            rng,
            excitatory_weight=0.18,
            inhibitory_weight=-0.12,
            inhibitory_fraction=0.15,
        )

        # A learnable, explicit sensory projection makes the first experiment
        # inspectable: input features can influence every action population.
        action_count = self.layout.action_count
        source = np.repeat(np.arange(self.config.input_features), action_count)
        target = np.tile(np.arange(action.start, chars.start), self.config.input_features)
        weights = rng.normal(0.9, 0.08, source.size)
        synapses.add_edges(source, target, weights)

        affect_source = np.repeat(np.arange(self.config.input_features), affect_count)
        affect_target = np.tile(
            np.arange(affect.start, action.start), self.config.input_features
        )
        affect_edge_start = synapses.weight.size
        synapses.add_edges(
            affect_source,
            affect_target,
            rng.normal(0.0, 0.02, affect_source.size),
        )
        self.affect_edge_indices = (
            AffectiveCircuit(self.config.neurons_per_affect).projection_indices(
                self.layout
            )
            + affect_edge_start
        )

        # Affective populations influence action populations through ordinary
        # recurrent synapses. There is no host-side affect-to-action arithmetic.
        affect_to_action_source = np.repeat(
            np.arange(affect.start, action.start), action_count
        )
        affect_to_action_target = np.tile(
            np.arange(action.start, chars.start), affect_count
        )
        affect_action_edge_start = synapses.weight.size
        synapses.add_edges(
            affect_to_action_source,
            affect_to_action_target,
            rng.normal(0.0, 0.02, affect_to_action_source.size),
        )
        self.affect_action_edge_indices = np.arange(
            affect_action_edge_start, synapses.weight.size
        )

        token_source_count = chars.start
        token_source = np.repeat(np.arange(token_source_count), self.language.neuron_count)
        token_target = np.tile(
            np.arange(chars.start, neuron_count), token_source_count
        )
        token_edge_start = synapses.weight.size
        synapses.add_edges(
            token_source,
            token_target,
            rng.normal(0.0, 0.025, token_source.size),
        )
        token_edge_indices = self.language.token_projection_indices(self.layout)
        self.token_input_edge_indices = token_edge_indices[: self.config.input_features] + token_edge_start
        hidden = self.layout.slice(Population.HIDDEN)
        recurrent_sources = (
            synapses.source >= hidden.start
        )
        recurrent_targets = synapses.target >= chars.start
        self.recurrent_event_edge_indices = np.flatnonzero(
            recurrent_sources & recurrent_targets
        )

        self.network = SpikingNetwork(neurons, synapses)
        self.encoder = TextEncoder(
            self.config.input_features,
            presentation_speed=self.config.presentation_speed,
        )
        self.working_memory = WorkingMemory(self.config.input_features)
        self.readout = ActionReadout(neurons_per_action=self.config.neurons_per_action)
        self.output_readout = EventReadout(self.layout)
        self.plasticity = RewardModulatedSTDP(synapses, learning_rate=0.08)
        self.affect = AffectiveState()
        self.affect_circuit = AffectiveCircuit(
            neurons_per_signal=self.config.neurons_per_affect,
        )
        self.reward_baseline = 0.0
        # Accounted stream rewards are kept separately from scalar metrics so
        # plasticity can consume the same delayed outcome that callers inspect.
        self.reward_ledger: list[AccountedEvent] = []
        self.last_snapshot: DebugSnapshot | None = None
        self.last_execution_snapshot: SessionExecutionSnapshot | None = None
        self.response_session = ResponseSession(
            policy=SessionPolicy(
                reset_neuron_state=not self.config.persistent_working_memory,
                reset_synaptic_activity=not self.config.persistent_working_memory,
                persist_working_memory=self.config.persistent_working_memory,
            )
        )

    def _start_response_session(self) -> None:
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
        if not policy.persist_affect:
            self.affect.reset()
        if not policy.persist_working_memory:
            self.working_memory.reset()
        if not policy.persist_plasticity_eligibility:
            self.plasticity.reset_traces()
        if policy.reset_readout:
            self.output_readout.reset()
        state = self.network.state_snapshot()
        voltage = state["voltage"]
        refractory = state["refractory"]
        pending_current = state["pending_current"]
        assert isinstance(voltage, np.ndarray)
        assert isinstance(refractory, np.ndarray)
        assert isinstance(pending_current, np.ndarray)
        neural = SessionSnapshot(
            voltage,
            refractory,
            pending_current,
            affect=self.affect,
            working_memory=self.working_memory,
            plasticity_eligibility=self.plasticity.eligibility,
        )
        self.last_execution_snapshot = SessionExecutionSnapshot(
            neural, self.network.synapses.weight
        )
        self.response_session.begin(self.last_execution_snapshot.neural)

    def _current_session_snapshot(self) -> SessionSnapshot:
        state = self.network.state_snapshot()
        voltage = state["voltage"]
        refractory = state["refractory"]
        pending_current = state["pending_current"]
        assert isinstance(voltage, np.ndarray)
        assert isinstance(refractory, np.ndarray)
        assert isinstance(pending_current, np.ndarray)
        return SessionSnapshot(
            voltage,
            refractory,
            pending_current,
            affect=self.affect,
            working_memory=self.working_memory,
            plasticity_eligibility=self.plasticity.eligibility,
        )

    def _finish_response(self, *, exhausted: bool) -> None:
        snapshot = self._current_session_snapshot()
        self.last_execution_snapshot = SessionExecutionSnapshot(
            snapshot, self.network.synapses.weight
        )
        if exhausted:
            self.response_session.exhaust(snapshot)
        else:
            self.response_session.receive_eos()
            self.response_session.complete(snapshot)

    def _frame_current(self, frame: np.ndarray) -> np.ndarray:
        current = np.zeros(self.network.neurons.count)
        current[: self.config.input_features] = frame
        if self.response_session.policy.persist_working_memory:
            current[: self.config.input_features] += self.working_memory.context_current()
        # Keep spontaneous internal affect activity, while candidate outputs
        # are driven only by network evidence or teacher forcing.
        affect = self.layout.slice(Population.AFFECT)
        current[affect] = 1.01
        return current

    def respond(self, text: str) -> Decision:
        """Process a message and deliberate until a decision or timeout."""

        self._start_response_session()
        self.affect.advance()
        spikes: list[np.ndarray] = []
        presentation = self.encoder.present(text)
        feature_activity = self.encoder.feature_vector(text)
        if self.response_session.policy.persist_working_memory:
            self.working_memory.update(feature_activity)
        for event in presentation:
            if event.signal is not None:
                self.response_session.handle_input_signal(event.signal)
                continue
            assert event.frame is not None
            self.response_session.accept_input_frame()
            emitted = self.network.step(self._frame_current(event.frame))
            self.plasticity.observe(emitted)
            spikes.append(emitted)

        self.response_session.begin_response()
        blank = np.zeros(self.config.input_features)
        action_event = None
        for _ in range(self.config.max_thinking_ticks):
            emitted = self.network.step(self._frame_current(blank))
            self.plasticity.observe(emitted)
            spikes.append(emitted)
            action_event = self.output_readout.observe(
                emitted,
                activation=self.network.neurons.voltage,
                populations=(Population.OUTPUT_ACTION,),
            )
            if action_event is not None:
                break
        if action_event is None:
            decision = Decision(
                action=Action.WAIT,
                confidence=0.0,
                ticks=len(spikes),
                evidence={action: 0.0 for action in self.actions},
                timed_out=True,
            )
        else:
            selected_action = Action(action_event.name)
            decision = Decision(
                action=selected_action,
                confidence=1.0,
                ticks=len(spikes),
                evidence={
                    action: action_event.evidence if action is selected_action else 0.0
                    for action in self.actions
                },
                timed_out=False,
            )
        neural_affect = self.affect_circuit.decode(spikes, self.layout)
        result = decision
        self.last_snapshot = DebugSnapshot.from_decision(
            result,
            self.affect,
            neural_affect,
            self.working_memory.snapshot(),
        )
        # An action event is complete host-facing output, but is not a
        # character EOS event. Do not manufacture EOS in the session state.
        self.response_session.abort(self._current_session_snapshot())
        return result

    def execute_isolated(
        self,
        operation: Callable[["ConversationAgent"], T],
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
    ) -> tuple[T, SessionExecutionSnapshot]:
        """Run an operation on copied agent state and optionally merge weight deltas.

        The copied agent owns neurons, recurrent activity, traces, readouts, and
        weights for the duration of ``operation``. By default no state is merged
        back; ``merge=True`` explicitly applies validated sparse weight deltas.
        """

        baseline = self._current_session_snapshot()
        execution = SessionExecutionSnapshot(baseline, self.network.synapses.weight)
        # Construct the shell normally, then copy only runtime state.  A full
        # deepcopy also reaches immutable layout mapping proxies on Python 3.14.
        isolated = ConversationAgent(self.config)
        isolated.network = self.network.copy()
        isolated.plasticity = deepcopy(self.plasticity)
        isolated.plasticity.synapses = isolated.network.synapses
        isolated.working_memory = deepcopy(self.working_memory)
        isolated.affect = deepcopy(self.affect)
        isolated.output_readout = EventReadout(
            self.layout,
            threshold=self.output_readout.threshold,
            cooldown=self.output_readout.cooldown,
            arbitration=deepcopy(self.output_readout.arbitration),
        )
        isolated.affect_circuit = deepcopy(self.affect_circuit)
        isolated.reward_baseline = self.reward_baseline
        isolated.reward_ledger = list(self.reward_ledger)
        isolated.last_snapshot = None
        isolated.response_session = ResponseSession(policy=self.response_session.policy)
        isolated.network.restore_state(self.network.state_snapshot())
        isolated.network.synapses.weight[:] = execution.weights

        result = operation(isolated)
        changed = np.flatnonzero(
            ~np.isclose(
                isolated.network.synapses.weight,
                execution.weights,
                rtol=0.0,
                atol=0.0,
            )
        )
        if changed.size:
            execution.record_weight_update(
                changed, isolated.network.synapses.weight[changed]
            )
        if merge:
            execution.merge_into(self.network.synapses.weight, conflict_policy)
        return result, execution

    def respond_isolated(
        self,
        text: str,
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
    ) -> tuple[Decision, SessionExecutionSnapshot]:
        """Respond without mutating this agent except for an explicit weight merge."""

        return self.execute_isolated(
            lambda isolated: isolated.respond(text),
            conflict_policy=conflict_policy,
            merge=merge,
        )

    def train_response_isolated(
        self,
        text: str,
        expected: Action,
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
    ) -> tuple[AgentAction, SessionExecutionSnapshot]:
        """Train on copied state and return the resulting explicit weight delta."""

        return self.execute_isolated(
            lambda isolated: isolated.train_response(text, expected),
            conflict_policy=conflict_policy,
            merge=merge,
        )

    def generate_response(self, act: Action, max_tokens: int | None = None) -> GeneratedResponse:
        """Generate externally visible character events from the network.

        ``EventReadout`` is authoritative here: silence produces no token and
        EOS is a discrete terminal event.
        """

        limit = self.config.max_response_tokens if max_tokens is None else max_tokens
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("max_tokens must be an integer or None")
        if limit < 0:
            raise ValueError("max_tokens cannot be negative")
        if self.config.max_response_ticks < 0:
            raise ValueError("max_response_ticks cannot be negative")
        self._start_response_session()
        self.response_session.begin_response()
        generated: list[str] = []
        context_features = self.encoder.feature_vector(act.value)
        emitted_tokens = 0
        # A token limit bounds externally emitted events, not recurrent ticks.
        # The network is allowed to be silent between events without inventing
        # output from a fixed observation window.
        for tick in range(self.config.max_response_ticks):
            if emitted_tokens >= limit:
                break
            current = self._frame_current(
                context_features if tick == 0 else np.zeros(self.config.input_features)
            )
            emitted = self.network.step(current)
            self.plasticity.observe(emitted)
            event = self.output_readout.observe(
                emitted,
                activation=self.network.neurons.voltage,
                populations=(Population.OUTPUT_CHAR,),
            )
            if event is None:
                continue
            emitted_tokens += 1
            token = event.name
            if token == "<EOS>":
                self._finish_response(exhausted=False)
                return GeneratedResponse(
                    "".join(generated), tuple(generated), True, len(generated) + 1
                )
            generated.append(token)
        self._finish_response(exhausted=True)
        return GeneratedResponse(
            "".join(generated), tuple(generated), False, len(generated)
        )

    def train_response_events(self, act: Action, target_events: Iterable[str]) -> None:
        """Teacher-guide an ordered event stream through existing recurrence.

        The action context is presented once through ``INPUT``.  Each target is
        then presented on the existing character output population while the
        network advances, allowing hidden and prior character activity to train
        the next event without feeding decoded characters back as input.
        """

        context = self.encoder.feature_vector(act.value)
        blank = np.zeros(self.config.input_features)
        for event_index, event in enumerate(target_events):
            token = getattr(event, "label", event)
            current = self._frame_current(context if event_index == 0 else blank)
            emitted = self.network.step(current)
            source_activity = np.maximum(
                self.network.neurons.voltage,
                emitted.astype(float),
            )
            if event_index == 0:
                self.language.align_next_token(
                    self.network.synapses,
                    self.token_input_edge_indices,
                    context,
                    token,
                )
            self.language.align_recurrent_token(
                self.network.synapses,
                self.recurrent_event_edge_indices,
                source_activity,
                token,
                layout=self.layout,
            )

            # Teacher forcing is an event presentation, not an external input
            # or a generated-character feedback connection.
            teacher = self._frame_current(blank)
            teacher[self.layout.subgroup(Population.OUTPUT_CHAR, token)] += 3.0
            self.network.step(teacher)

        self.plasticity.reset_traces()

    def apply_event_stream_reward(
        self,
        report: EventStreamReport,
        *,
        target_neurons: np.ndarray | None = None,
    ) -> float:
        """Commit an evaluated stream reward through the plasticity path.

        ``evaluate_event_stream`` remains responsible for timing tolerance and
        outcome classification.  This method records every resulting reward,
        then applies their sum once, after the stream's eligibility has been
        accumulated.  It returns the reward prediction error used by STDP.
        """
        self.reward_ledger.extend(report.records)
        prediction_error = report.total_reward - self.reward_baseline
        self.reward_baseline += 0.05 * prediction_error
        self.plasticity.reinforce(
            prediction_error,
            self.affect.modulation(),
            target_neurons=target_neurons,
        )
        self.plasticity.reset_traces()
        return prediction_error

    def train_response_stream(
        self,
        act: Action,
        targets: Iterable[TargetEvent | str],
        *,
        config: EventStreamConfig | None = None,
        end_time: int | None = None,
        silence_intervals: Iterable[SilenceInterval] = (),
    ) -> EventStreamReport:
        """Generate, evaluate, and reinforce one character event stream."""
        from continual_agent.evaluation.event_stream import evaluate_event_stream

        target_events = tuple(targets)
        self.generate_response(act)
        report = evaluate_event_stream(
            target_events,
            tuple(self.output_readout.events),
            config=config,
            end_time=self.config.max_response_ticks if end_time is None else end_time,
            silence_intervals=silence_intervals,
        )
        self.apply_event_stream_reward(report)
        return report

    def train_response(self, text: str, expected: Action) -> AgentAction:
        decision = self.respond(text)
        selected_action = decision.action
        reward = 1.0 if selected_action == expected else -1.0
        prediction_error = reward - self.reward_baseline
        self.reward_baseline += 0.05 * prediction_error
        action_group = self.readout.groups(self.layout)[selected_action]
        self.plasticity.reinforce(
            prediction_error,
            self.affect.modulation(),
            target_neurons=action_group,
        )
        self.plasticity.reset_traces()
        self.affect.observe(
            AffectiveEvent(
                reward_prediction_error=prediction_error,
                uncertainty=0.8 if decision.timed_out else 0.2,
                correction=expected == Action.REVISE,
                social_feedback=1.0 if expected == Action.ACKNOWLEDGE else 0.0,
            )
        )
        self.affect_circuit.align(
            self.network.synapses,
            self.affect_edge_indices,
            self.encoder.feature_vector(text),
            self.affect,
        )
        self.last_snapshot = DebugSnapshot.from_decision(
            Decision(
                action=selected_action,
                confidence=decision.confidence,
                ticks=decision.ticks,
                evidence=decision.evidence,
                timed_out=decision.timed_out,
            ),
            self.affect,
            self.affect_circuit.projection_prediction(
                self.network.synapses,
                self.affect_edge_indices,
                self.encoder.feature_vector(text),
            ),
        )
        return AgentAction(selected_action, decision.confidence, decision.ticks)

    def reset_conversation(self) -> None:
        """Clear turn context while retaining learned synapses and affect priors."""

        self.working_memory.reset()
        self.network.reset_state()

    def debug_snapshot(self) -> dict[str, object] | None:
        """Return the latest state snapshot for logs or an optional UI."""

        return self.last_snapshot.as_dict() if self.last_snapshot else None
