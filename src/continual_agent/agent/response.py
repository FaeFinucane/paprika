"""Response lifecycle and output generation for the conversational adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from continual_agent.agent.debug import DebugSnapshot
from continual_agent.agent.session import InputSignal, SessionExecutionSnapshot, SessionSnapshot
from continual_agent.cognition.readout import Action, Decision
from continual_agent.language.spiking_decoder import GeneratedResponse
from continual_agent.simulation.population_layout import Population

if TYPE_CHECKING:
    pass


class ResponseMixin:
    last_snapshot: DebugSnapshot | None
    last_execution_snapshot: SessionExecutionSnapshot | None

    def _start_response_session(self: Any) -> None:
        policy = self.response_session.policy
        if not policy.persist_affect:
            self.affect.reset()
        if not policy.persist_working_memory:
            self.working_memory.reset()
        self.runtime.start_session(affect=self.affect, working_memory=self.working_memory)
        snapshot = self.runtime.response_session.snapshot
        assert snapshot is not None
        self.last_execution_snapshot = SessionExecutionSnapshot(
            snapshot, self.network.synapses.weight
        )

    def _current_session_snapshot(self: Any) -> SessionSnapshot:
        return self.runtime._snapshot(self.affect, self.working_memory)

    def _finish_response(self: Any, *, exhausted: bool) -> None:
        snapshot = self._current_session_snapshot()
        self.last_execution_snapshot = SessionExecutionSnapshot(
            snapshot, self.network.synapses.weight
        )
        self.runtime.finish_session(
            exhausted=exhausted, affect=self.affect, working_memory=self.working_memory
        )

    def _frame_current(self: Any, frame: np.ndarray) -> np.ndarray:
        current = self.runtime.current(frame, tonic_affect=True)
        if self.response_session.policy.persist_working_memory:
            current[: self.config.input_features] += self.working_memory.context_current()
        return current

    def respond(self: Any, text: str) -> Decision:
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
                self.runtime.step_input_signal(event.signal)
                if event.signal is InputSignal.INPUT_END:
                    self.response_session.begin_response()
                continue
            assert event.frame is not None
            self.response_session.accept_input_frame()
            spikes.append(self.runtime.step(self._frame_current(event.frame)))

        if self.response_session.state.value != "responding":
            self.response_session.begin_response()
        blank = np.zeros(self.config.input_features)
        action_event = None
        for _ in range(self.config.max_thinking_ticks):
            emitted = self.runtime.step(self._frame_current(blank))
            spikes.append(emitted)
            action_event = self.output_readout.observe(
                emitted,
                activation=self.network.neurons.voltage,
                populations=(Population.OUTPUT_ACTION,),
            )
            if action_event is not None:
                break
        if action_event is None:
            result = Decision(
                action=Action.WAIT,
                confidence=0.0,
                ticks=len(spikes),
                evidence={action: 0.0 for action in self.actions},
                timed_out=True,
            )
        else:
            selected_action = Action(action_event.name)
            result = Decision(
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
        self.last_snapshot = DebugSnapshot.from_decision(
            result, self.affect, neural_affect, self.working_memory.snapshot()
        )
        self.response_session.abort(self._current_session_snapshot())
        return result

    def generate_response(
        self: Any, act: Action, max_tokens: int | None = None
    ) -> GeneratedResponse:
        """Generate character events; silence is not an event."""
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
        context = self.encoder.feature_vector(act.value)
        emitted_tokens = 0
        for tick in range(self.config.max_response_ticks):
            if emitted_tokens >= limit:
                break
            emitted = self.runtime.step(
                self._frame_current(context if tick == 0 else np.zeros(self.config.input_features))
            )
            event = self.output_readout.observe(
                emitted,
                activation=self.network.neurons.voltage,
                populations=(Population.OUTPUT_CHAR,),
            )
            if event is None:
                continue
            emitted_tokens += 1
            if event.name == "<EOS>":
                self._finish_response(exhausted=False)
                return GeneratedResponse(
                    "".join(generated), tuple(generated), True, len(generated) + 1
                )
            generated.append(event.name)
        self._finish_response(exhausted=True)
        return GeneratedResponse("".join(generated), tuple(generated), False, len(generated))
