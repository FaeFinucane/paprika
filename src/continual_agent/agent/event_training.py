"""Raw and teacher-guided event training for the conversational adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from continual_agent.agent.session import InputSignal
from continual_agent.simulation.population_layout import Population

if TYPE_CHECKING:
    from continual_agent.cognition.readout import Action, OutputEvent
    from continual_agent.evaluation.event_stream import (
        EventStreamConfig,
        EventStreamReport,
        SilenceInterval,
        TargetEvent,
    )


class EventTrainingMixin:
    def train_response_events(self: Any, act: Action, target_events: Iterable[str]) -> None:
        context = self.encoder.feature_vector(act.value)
        blank = np.zeros(self.config.input_features)
        for event_index, event in enumerate(target_events):
            token = getattr(event, "label", event)
            current = self._frame_current(context if event_index == 0 else blank)
            emitted = self.runtime.step(current)
            source_activity = np.maximum(self.network.neurons.voltage, emitted.astype(float))
            if event_index == 0:
                self.language.align_next_token(
                    self.network.synapses, self.token_input_edge_indices, context, token
                )
            self.language.align_recurrent_token(
                self.network.synapses,
                self.recurrent_event_edge_indices,
                source_activity,
                token,
                layout=self.layout,
            )
            teacher = self._frame_current(blank)
            teacher[self.layout.subgroup(Population.OUTPUT_CHAR, token)] += 3.0
            self.runtime.step(teacher)
        self.plasticity.reset_traces()

    def train_input_events(
        self: Any, events: Iterable[InputSignal | tuple[np.ndarray, str]]
    ) -> None:
        self.runtime.train_input_events(events, current_builder=self._frame_current)

    def run_input_events(
        self: Any,
        events: Iterable[InputSignal | np.ndarray],
        *,
        response_ticks: int,
        observe_during_input: bool = True,
    ) -> tuple[OutputEvent, ...]:
        return self.runtime.run_input_events(
            events,
            response_ticks=response_ticks,
            observe_during_input=observe_during_input,
            current_builder=self._frame_current,
        )

    def train_response_stream(
        self: Any,
        act: Action,
        targets: Iterable[TargetEvent | str],
        *,
        config: EventStreamConfig | None = None,
        end_time: int | None = None,
        silence_intervals: Iterable[SilenceInterval] = (),
    ) -> EventStreamReport:
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
