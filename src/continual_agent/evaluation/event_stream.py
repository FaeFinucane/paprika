"""Tolerant accounting and reward semantics for discrete output streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from continual_agent.cognition.readout import OutputEvent


class EventOutcome(str, Enum):
    CORRECT_EVENT = "correct_event"
    INCORRECT_EVENT = "incorrect_event"
    MISSING_OUTPUT = "missing_output"
    VALID_SILENCE = "valid_silence"
    UNWANTED_OUTPUT = "unwanted_output"
    PREMATURE_EOS = "premature_eos"
    MISSING_EOS = "missing_eos"
    POST_EOS_OUTPUT = "post_eos_output"


@dataclass(frozen=True)
class RewardSchedule:
    """Event-level third-factor schedule; penalties are not volume-scaled."""

    stage: str = "early"
    correct: float = 2.0
    incorrect: float = -0.05
    missing: float = 0.0
    silence: float = 0.0
    unwanted: float = -0.05
    premature_eos: float = -0.05
    missing_eos: float = 0.0
    post_eos: float = -0.05

    @classmethod
    def for_stage(cls, stage: str) -> "RewardSchedule":
        if stage == "early":
            return cls(stage)
        if stage == "reliable":
            return cls(stage, 2.0, -0.5, -0.25, 0.0, -0.5, -0.5, -0.25, -0.5)
        raise ValueError("reward stage must be 'early' or 'reliable'")

    def event_config(self, *, patience_window: int) -> "EventStreamConfig":
        return EventStreamConfig(
            patience_window=patience_window,
            correct_reward=self.correct,
            incorrect_reward=self.incorrect,
            missing_reward=self.missing,
            silence_reward=self.silence,
            unwanted_reward=self.unwanted,
            premature_eos_reward=self.premature_eos,
            missing_eos_reward=self.missing_eos,
            post_eos_reward=self.post_eos,
        )


@dataclass(frozen=True)
class TargetEvent:
    """An expected event.  Timestamps are optional and never required."""

    label: str
    timestamp: int | None = None


@dataclass(frozen=True)
class SilenceInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("silence interval start must not exceed end")


@dataclass(frozen=True)
class EventStreamConfig:
    """Scoring policy; early timing is open unless explicitly constrained."""

    patience_window: int = 8
    early_tolerance: int | None = None
    measure_latency: bool = True
    measure_timing_error: bool = False
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    missing_reward: float = -1.0
    silence_reward: float = 0.0
    unwanted_reward: float = -1.0
    premature_eos_reward: float = -1.0
    missing_eos_reward: float = -1.0
    post_eos_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.patience_window < 0 or (
            self.early_tolerance is not None and self.early_tolerance < 0
        ):
            raise ValueError("timing windows must be non-negative")


@dataclass(frozen=True)
class AccountedEvent:
    outcome: EventOutcome
    label: str | None
    timestamp: int | None
    reward: float
    latency: int | None = None
    timing_error: int | None = None


@dataclass(frozen=True)
class _ObservedEvent:
    name: str
    timestamp: int


@dataclass
class EventStreamReport:
    records: list[AccountedEvent] = field(default_factory=list)
    silence_duration: int = 0
    eos_accuracy: bool = False
    premature_eos: int = 0
    post_eos_output: int = 0

    @property
    def rewards(self) -> list[float]:
        return [record.reward for record in self.records]

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def counts(self) -> dict[EventOutcome, int]:
        return {
            outcome: sum(record.outcome is outcome for record in self.records)
            for outcome in EventOutcome
        }

    @property
    def event_precision(self) -> float:
        emitted = sum(
            record.outcome
            in {
                EventOutcome.CORRECT_EVENT,
                EventOutcome.INCORRECT_EVENT,
                EventOutcome.UNWANTED_OUTPUT,
                EventOutcome.PREMATURE_EOS,
            }
            for record in self.records
        )
        return self.counts[EventOutcome.CORRECT_EVENT] / emitted if emitted else 0.0

    @property
    def event_recall(self) -> float:
        expected = sum(
            record.outcome
            in {
                EventOutcome.CORRECT_EVENT,
                EventOutcome.MISSING_OUTPUT,
                EventOutcome.MISSING_EOS,
            }
            for record in self.records
        )
        return self.counts[EventOutcome.CORRECT_EVENT] / expected if expected else 0.0

    @property
    def response_latency(self) -> int | None:
        values = [
            record.timestamp
            for record in self.records
            if record.outcome is EventOutcome.CORRECT_EVENT and record.timestamp is not None
        ]
        return min(values) if values else None

    @property
    def mean_latency(self) -> float | None:
        values = [record.latency for record in self.records if record.latency is not None]
        return sum(values) / len(values) if values else None

    @property
    def latencies(self) -> list[int]:
        return [record.latency for record in self.records if record.latency is not None]

    @property
    def timing_errors(self) -> list[int]:
        return [record.timing_error for record in self.records if record.timing_error is not None]


def _coerce_target(target: TargetEvent | str) -> TargetEvent:
    return target if isinstance(target, TargetEvent) else TargetEvent(str(target))


def evaluate_event_stream(
    targets: Iterable[TargetEvent | str],
    observed: Iterable[OutputEvent | str],
    *,
    config: EventStreamConfig | None = None,
    end_time: int | None = None,
    silence_intervals: Iterable[SilenceInterval] = (),
    withheld_prefix: int = 0,
) -> EventStreamReport:
    """Account for an ordered stream without imposing fixed inter-event timing.

    ``end_time`` closes the final patience window. Without it, an unfinished
    target is not declared missing merely because the caller stopped observing.
    ``withheld_prefix`` accounts for leading targets whose output was
    intentionally unavailable during input presentation.
    """
    config = config or EventStreamConfig()
    expected = [_coerce_target(target) for target in targets]
    events = [
        _ObservedEvent(event.name, event.timestamp)
        if isinstance(event, OutputEvent)
        else _ObservedEvent(str(event), index)
        for index, event in enumerate(observed)
    ]
    target_timestamps = [target.timestamp for target in expected if target.timestamp is not None]
    if any(timestamp < 0 for timestamp in target_timestamps):
        raise ValueError("target timestamps must be non-negative")
    if any(left >= right for left, right in zip(target_timestamps, target_timestamps[1:])):
        raise ValueError("target timestamps must increase")
    observed_timestamps = [event.timestamp for event in events]
    if any(timestamp < 0 for timestamp in observed_timestamps):
        raise ValueError("observed timestamps must be non-negative")
    if any(left >= right for left, right in zip(observed_timestamps, observed_timestamps[1:])):
        raise ValueError("observed timestamps must increase")
    if end_time is not None and end_time < 0:
        raise ValueError("end_time must be non-negative")
    if withheld_prefix < 0 or withheld_prefix > len(expected):
        raise ValueError("withheld_prefix must be within the target stream")
    if any(target.label == "<EOS>" for target in expected[:withheld_prefix]):
        raise ValueError("withheld_prefix must not include EOS")
    report = EventStreamReport()
    silence = tuple(silence_intervals)
    position = withheld_prefix
    eos_seen = False
    last_timestamp: int | None = None

    def add(
        outcome: EventOutcome,
        label: str | None,
        timestamp: int | None,
        reward: float,
        latency: int | None = None,
        timing_error: int | None = None,
    ) -> None:
        report.records.append(
            AccountedEvent(outcome, label, timestamp, reward, latency, timing_error)
        )

    for target in expected[:withheld_prefix]:
        add(EventOutcome.MISSING_OUTPUT, target.label, target.timestamp, config.missing_reward)

    for event in events:
        if last_timestamp is None:
            report.silence_duration += max(0, event.timestamp)
        if last_timestamp is not None and event.timestamp > last_timestamp:
            report.silence_duration += event.timestamp - last_timestamp - 1
        last_timestamp = event.timestamp
        if eos_seen:
            report.post_eos_output += 1
            add(EventOutcome.POST_EOS_OUTPUT, event.name, event.timestamp, config.post_eos_reward)
            continue
        while position < len(expected):
            target = expected[position]
            if (
                target.timestamp is None
                or event.timestamp <= target.timestamp + config.patience_window
            ):
                break
            outcome = (
                EventOutcome.MISSING_EOS if target.label == "<EOS>" else EventOutcome.MISSING_OUTPUT
            )
            reward = (
                config.missing_eos_reward
                if outcome is EventOutcome.MISSING_EOS
                else config.missing_reward
            )
            add(outcome, target.label, target.timestamp, reward)
            position += 1
        if position >= len(expected):
            add(EventOutcome.UNWANTED_OUTPUT, event.name, event.timestamp, config.unwanted_reward)
            continue
        target = expected[position]
        if any(interval.start <= event.timestamp <= interval.end for interval in silence):
            add(EventOutcome.UNWANTED_OUTPUT, event.name, event.timestamp, config.unwanted_reward)
            continue
        if event.name == "<EOS>" and target.label != "<EOS>":
            report.premature_eos += 1
            add(
                EventOutcome.PREMATURE_EOS, event.name, event.timestamp, config.premature_eos_reward
            )
            eos_seen = True
            continue
        early_ok = (
            config.early_tolerance is None
            or target.timestamp is None
            or event.timestamp >= target.timestamp - config.early_tolerance
        )
        if event.name == target.label and early_ok:
            latency = (
                event.timestamp - target.timestamp
                if target.timestamp is not None and config.measure_latency
                else None
            )
            timing = (
                event.timestamp - target.timestamp
                if target.timestamp is not None and config.measure_timing_error
                else None
            )
            add(
                EventOutcome.CORRECT_EVENT,
                event.name,
                event.timestamp,
                config.correct_reward,
                latency,
                timing,
            )
            position += 1
            if target.label == "<EOS>":
                eos_seen = True
                report.eos_accuracy = True
        else:
            add(EventOutcome.INCORRECT_EVENT, event.name, event.timestamp, config.incorrect_reward)

    if end_time is not None:
        while position < len(expected):
            target = expected[position]
            deadline = (
                target.timestamp + config.patience_window
                if target.timestamp is not None
                else end_time
            )
            if deadline > end_time:
                break
            outcome = (
                EventOutcome.MISSING_EOS if target.label == "<EOS>" else EventOutcome.MISSING_OUTPUT
            )
            reward = (
                config.missing_eos_reward
                if outcome is EventOutcome.MISSING_EOS
                else config.missing_reward
            )
            add(outcome, target.label, deadline, reward)
            position += 1
    if (
        position < len(expected)
        and expected[position].label == "<EOS>"
        and not eos_seen
        and end_time is not None
    ):
        report.eos_accuracy = False
    if end_time is not None:
        report.silence_duration += max(
            0, end_time - (last_timestamp if last_timestamp is not None else -1)
        )
    if report.silence_duration:
        add(EventOutcome.VALID_SILENCE, None, None, config.silence_reward)
    return report
