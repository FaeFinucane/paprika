"""A text-only vertical slice around the spiking network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_agent.agent.debug import DebugSnapshot
from continual_agent.cognition.affect import AffectiveEvent, AffectiveState
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.cognition.readout import Action, ActionReadout, Decision
from continual_agent.cognition.working_memory import WorkingMemory
from continual_agent.encoding.text_encoder import TextEncoder
from continual_agent.environment.protocol import AgentAction
from continual_agent.language.spiking_decoder import GeneratedResponse, SpikingCharacterDecoder
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.neurons import LIFNeurons
from continual_agent.simulation.network import SpikingNetwork
from continual_agent.simulation.synapses import SparseSynapses


@dataclass(frozen=True)
class AgentConfig:
    input_features: int = 48
    neurons_per_action: int = 4
    neurons_per_affect: int = 4
    hidden_neurons: int = 48
    max_thinking_ticks: int = 18
    connection_probability: float = 0.08
    exploration_epsilon: float = 0.15
    seed: int = 0
    persistent_working_memory: bool = True
    neurons_per_token: int = 3
    max_response_tokens: int = 48
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
        affect_count = len(AffectiveCircuit.signal_names) * self.config.neurons_per_affect
        self.affect_start = self.config.input_features + self.config.hidden_neurons
        self.output_start = self.affect_start + affect_count
        output_count = len(self.actions) * self.config.neurons_per_action
        self.token_start = self.output_start + output_count
        neuron_count = self.token_start + self.language.neuron_count
        rng = np.random.default_rng(self.config.seed)
        self.rng = rng
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
        action_count = len(self.actions) * self.config.neurons_per_action
        source = np.repeat(np.arange(self.config.input_features), action_count)
        target = np.tile(np.arange(self.output_start, self.token_start), self.config.input_features)
        weights = rng.normal(0.0, 0.08, source.size)
        synapses.add_edges(source, target, weights)

        affect_source = np.repeat(np.arange(self.config.input_features), affect_count)
        affect_target = np.tile(
            np.arange(self.affect_start, self.output_start), self.config.input_features
        )
        affect_edge_start = synapses.weight.size
        synapses.add_edges(
            affect_source,
            affect_target,
            rng.normal(0.0, 0.02, affect_source.size),
        )
        self.affect_edge_indices = (
            AffectiveCircuit(self.config.neurons_per_affect).projection_indices(
                self.config.input_features, self.affect_start
            )
            + affect_edge_start
        )

        # Affective populations influence action populations through ordinary
        # recurrent synapses. There is no host-side affect-to-action arithmetic.
        affect_to_action_source = np.repeat(
            np.arange(self.affect_start, self.output_start), action_count
        )
        affect_to_action_target = np.tile(
            np.arange(self.output_start, self.token_start), affect_count
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

        token_source_count = self.token_start
        token_source = np.repeat(np.arange(token_source_count), self.language.neuron_count)
        token_target = np.tile(
            np.arange(self.token_start, neuron_count), token_source_count
        )
        token_edge_start = synapses.weight.size
        synapses.add_edges(
            token_source,
            token_target,
            rng.normal(0.0, 0.025, token_source.size),
        )
        self.token_edge_indices = (
            self.language.token_projection_indices(
                token_source_count, 0, self.token_start
            )
            + token_edge_start
        )
        self.token_input_edge_indices = self.token_edge_indices[: self.config.input_features]

        self.network = SpikingNetwork(neurons, synapses)
        self.encoder = TextEncoder(self.config.input_features)
        self.working_memory = WorkingMemory(self.config.input_features)
        self.readout = ActionReadout(neurons_per_action=self.config.neurons_per_action)
        self.plasticity = RewardModulatedSTDP(synapses, learning_rate=0.08)
        self.affect = AffectiveState()
        self.affect_circuit = AffectiveCircuit(
            neurons_per_signal=self.config.neurons_per_affect,
        )
        # Keep the old public name as a compatibility alias while callers move
        # to the richer affective state.
        self.drives = self.affect
        self.reward_baseline = 0.0
        self.last_snapshot: DebugSnapshot | None = None

    def _frame_current(self, frame: np.ndarray) -> np.ndarray:
        current = np.zeros(self.network.neurons.count)
        current[: self.config.input_features] = frame
        if self.config.persistent_working_memory:
            current[: self.config.input_features] += self.working_memory.context_current()
        output_indices = np.arange(self.affect_start, self.network.neurons.count)
        # A near-threshold tonic current keeps output populations in a useful
        # competitive regime. Sensory projections then alter spike timing and
        # counts instead of leaving every readout permanently silent.
        current[output_indices] = 1.01
        return current

    def respond(self, text: str) -> Decision:
        """Process a message and deliberate until a decision or timeout."""

        if not self.config.persistent_working_memory:
            self.network.reset_state()
        self.affect.advance()
        spikes: list[np.ndarray] = []
        frames = self.encoder.encode(text)
        feature_activity = self.encoder.feature_vector(text)
        if self.config.persistent_working_memory:
            self.working_memory.update(feature_activity)
        for frame in frames:
            emitted = self.network.step(self._frame_current(frame))
            self.plasticity.observe(emitted)
            spikes.append(emitted)

        blank = np.zeros(self.config.input_features)
        for _ in range(self.config.max_thinking_ticks):
            emitted = self.network.step(self._frame_current(blank))
            self.plasticity.observe(emitted)
            spikes.append(emitted)
            decision = self.readout.decide(spikes, self.output_start)
        decision = self.readout.decide(spikes, self.output_start)
        policy_scores = self.readout.score_policy(feature_activity)
        neural_affect = self.affect_circuit.decode(spikes, self.affect_start)
        # Keep the neural population readout visible, while allowing the
        # transparent local policy to make the first vertical slice learnable.
        combined = {
            action: 0.001 * evidence + 2.0 * policy_scores[action]
            for action, evidence in decision.evidence.items()
        }
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        best, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(max(0.0, score - min(0.0, ranked[-1][1])) for _, score in ranked)
        result = Decision(
            action=best,
            confidence=max(0.0, best_score - second_score) / max(1.0, total),
            ticks=decision.ticks,
            evidence=combined,
            timed_out=decision.timed_out,
        )
        self.last_snapshot = DebugSnapshot.from_decision(
            result,
            self.affect,
            neural_affect,
            self.working_memory.snapshot(),
        )
        return result

    def generate_response(self, act: Action, max_tokens: int | None = None) -> GeneratedResponse:
        """Generate characters from persistent spiking state until EOS."""

        limit = max_tokens or self.config.max_response_tokens
        generated: list[str] = []
        context_features = self.encoder.feature_vector(act.value)
        generated_context = context_features.copy()
        for _ in range(limit):
            token_frames: list[np.ndarray] = []
            current = self._frame_current(generated_context)
            for _ in range(3):
                emitted = self.network.step(current)
                self.plasticity.observe(emitted)
                token_frames.append(emitted)
                current = self._frame_current(np.zeros(self.config.input_features))
            token = self.language.choose_non_eos(token_frames, self.token_start)
            if not generated:
                active = generated_context
                scores = self.network.synapses.weight[
                    self.token_input_edge_indices[:, self.language.tokens.index(token)]
                ].mean(axis=1) @ active
                token_scores = {
                    candidate: float(
                        self.network.synapses.weight[
                            self.token_input_edge_indices[:, index]
                        ].mean(axis=1)
                        @ active
                    )
                    for index, candidate in enumerate(self.language.tokens)
                    if candidate != "<EOS>"
                }
                token = max(token_scores, key=lambda candidate: token_scores[candidate])
            if token == "<EOS>":
                return GeneratedResponse(
                    "".join(generated), tuple(generated), True, len(generated) + 1
                )
            generated.append(token)
            generated_context = self.encoder.feature_vector(token)
        return GeneratedResponse(
            "".join(generated), tuple(generated), False, len(generated)
        )

    def train_response_text(self, act: Action, target_text: str) -> None:
        """Teacher-guide token populations using local synaptic alignment."""

        context = self.encoder.feature_vector(act.value)
        previous = np.zeros(self.network.neurons.count)
        for token in self.language.target_tokens(target_text):
            self.language.align_next_token(
                self.network.synapses,
                self.token_input_edge_indices,
                context,
                token,
            )
            token_index = self.language.tokens.index(token)
            token_group = np.arange(
                self.token_start + token_index * self.config.neurons_per_token,
                self.token_start + (token_index + 1) * self.config.neurons_per_token,
            )
            previous[token_group] = 1.0
            context = self.encoder.feature_vector(token)

        self.plasticity.reset_traces()

    def respond_with_working_memory(self, text: str) -> Decision:
        """Compatibility helper used by context-sensitive experiments."""

        return self.respond(text)

    def train_response(self, text: str, expected: Action) -> AgentAction:
        decision = self.respond(text)
        selected_action = decision.action
        if self.rng.random() < self.config.exploration_epsilon:
            selected_action = self.actions[
                int(self.rng.integers(0, len(self.actions)))
            ]
        reward = 1.0 if selected_action == expected else -1.0
        prediction_error = reward - self.reward_baseline
        self.reward_baseline += 0.05 * prediction_error
        action_group = self.readout.groups(self.output_start)[selected_action]
        self.plasticity.reinforce(
            prediction_error,
            self.drives.modulation(),
            target_neurons=action_group,
        )
        frames = self.encoder.encode(text)
        self.readout.reinforce_policy(
            selected_action,
            self.encoder.feature_vector(text),
            prediction_error,
            learning_rate=0.2,
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
