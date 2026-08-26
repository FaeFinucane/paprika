"""Canonical public facade for the conversational spiking agent."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

from continual_agent.agent.config import AgentConfig
from continual_agent.agent.debug import DebugSnapshot
from continual_agent.agent.event_training import EventTrainingMixin
from continual_agent.agent.response import ResponseMixin
from continual_agent.agent.session import (
    ConflictPolicy,
    SessionExecutionSnapshot,
    SessionPolicy,
)
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.cognition.affect import AffectiveEvent, AffectiveState
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.cognition.readout import Action, ActionReadout, Decision
from continual_agent.cognition.working_memory import WorkingMemory
from continual_agent.encoding.text_encoder import TextEncoder
from continual_agent.environment.protocol import AgentAction
from continual_agent.language.spiking_decoder import SpikingCharacterDecoder

if TYPE_CHECKING:
    from continual_agent.evaluation.event_stream import AccountedEvent, EventStreamReport

T = TypeVar("T")


class ConversationAgent(ResponseMixin, EventTrainingMixin):
    """Continuously plastic text agent with a small typed action space."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.actions = tuple(Action)
        self.language = SpikingCharacterDecoder(
            alphabet=self.config.language_alphabet,
            neurons_per_token=self.config.neurons_per_token,
        )
        self.runtime = SpikingRuntime(
            input_features=self.config.input_features,
            hidden_neurons=self.config.hidden_neurons,
            output_tokens=self.language.tokens,
            neurons_per_token=self.config.neurons_per_token,
            action_names=tuple(action.value for action in self.actions),
            neurons_per_action=self.config.neurons_per_action,
            affect_names=AffectiveCircuit.signal_names,
            neurons_per_affect=self.config.neurons_per_affect,
            connection_probability=self.config.connection_probability,
            seed=self.config.seed,
            session_policy=SessionPolicy(
                reset_neuron_state=not self.config.persistent_working_memory,
                reset_synaptic_activity=not self.config.persistent_working_memory,
                persist_working_memory=self.config.persistent_working_memory,
            ),
            homeostasis_enabled=self.config.homeostasis_enabled,
            homeostasis_target_rate=self.config.homeostasis_target_rate,
            homeostasis_strength=self.config.homeostasis_strength,
            homeostasis_update_interval=self.config.homeostasis_update_interval,
            homeostasis_max_current=self.config.homeostasis_max_current,
            homeostasis_populations=self.config.homeostasis_populations,
        )
        self.encoder = TextEncoder(
            self.config.input_features,
            presentation_speed=self.config.presentation_speed,
        )
        self.working_memory = WorkingMemory(self.config.input_features)
        self.readout = ActionReadout(neurons_per_action=self.config.neurons_per_action)
        self.affect = AffectiveState()
        self.affect_circuit = AffectiveCircuit(neurons_per_signal=self.config.neurons_per_affect)
        self.reward_baseline = 0.0
        self.reward_ledger: list[AccountedEvent] = []
        self.last_snapshot: DebugSnapshot | None = None
        self.last_execution_snapshot: SessionExecutionSnapshot | None = None

    def execute_isolated(
        self,
        operation: Callable[[ConversationAgent], T],
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
    ) -> tuple[T, SessionExecutionSnapshot]:
        """Run an operation on copied agent state and optionally merge weights."""
        isolated = ConversationAgent(self.config)

        def transfer(_: SpikingRuntime, runtime: SpikingRuntime) -> None:
            isolated.runtime = runtime
            isolated.working_memory = deepcopy(self.working_memory)
            isolated.affect = deepcopy(self.affect)
            isolated.affect_circuit = deepcopy(self.affect_circuit)
            isolated.reward_baseline = self.reward_baseline
            isolated.reward_ledger = list(self.reward_ledger)
            isolated.last_snapshot = None

        return self.runtime.execute_isolated(
            lambda _: operation(isolated),
            conflict_policy=conflict_policy,
            merge=merge,
            extension_hook=transfer,
        )

    def respond_isolated(
        self,
        text: str,
        *,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        merge: bool = False,
    ) -> tuple[Decision, SessionExecutionSnapshot]:
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
        return self.execute_isolated(
            lambda isolated: isolated.train_response(text, expected),
            conflict_policy=conflict_policy,
            merge=merge,
        )

    def apply_event_stream_reward(
        self, report: EventStreamReport, *, target_neurons: np.ndarray | None = None
    ) -> float:
        self.reward_ledger.extend(report.records)
        prediction_error = report.total_reward - self.reward_baseline
        self.reward_baseline += 0.05 * prediction_error
        self.runtime.plasticity.reinforce(
            prediction_error, self.affect.modulation(), target_neurons=target_neurons
        )
        self.runtime.plasticity.reset_traces()
        return prediction_error

    def train_response(self, text: str, expected: Action) -> AgentAction:
        decision = self.respond(text)
        reward = 1.0 if decision.action == expected else -1.0
        prediction_error = reward - self.reward_baseline
        self.reward_baseline += 0.05 * prediction_error
        action_group = self.readout.groups(self.runtime.layout)[decision.action]
        self.runtime.plasticity.reinforce(
            prediction_error, self.affect.modulation(), target_neurons=action_group
        )
        self.runtime.plasticity.reset_traces()
        self.affect.observe(
            AffectiveEvent(
                reward_prediction_error=prediction_error,
                uncertainty=0.8 if decision.timed_out else 0.2,
                correction=expected == Action.REVISE,
                social_feedback=1.0 if expected == Action.ACKNOWLEDGE else 0.0,
            )
        )
        features = self.encoder.feature_vector(text)
        self.affect_circuit.align(
            self.runtime.network.synapses,
            self.runtime.affect_edge_indices,
            features,
            self.affect,
        )
        self.last_snapshot = DebugSnapshot.from_decision(
            decision,
            self.affect,
            self.affect_circuit.projection_prediction(
                self.runtime.network.synapses, self.runtime.affect_edge_indices, features
            ),
        )
        return AgentAction(decision.action, decision.confidence, decision.ticks)

    def reset_conversation(self) -> None:
        """Clear turn context while retaining learned synapses and affect priors."""
        self.working_memory.reset()
        self.runtime.network.reset_state()

    def debug_snapshot(self) -> dict[str, object] | None:
        return self.last_snapshot.as_dict() if self.last_snapshot else None
