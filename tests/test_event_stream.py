import numpy as np
import pytest

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.cognition.readout import OutputEvent
from continual_agent.evaluation.event_stream import (
    EventOutcome,
    EventStreamConfig,
    EventStreamReport,
    RewardSchedule,
    SilenceInterval,
    TargetEvent,
    evaluate_event_stream,
)
from continual_agent.simulation.population_layout import Population


def test_early_reward_schedule_is_asymmetric_and_event_level() -> None:
    config = RewardSchedule.for_stage("early").event_config(patience_window=2)
    assert config.correct_reward > 1.0
    assert -0.1 <= config.incorrect_reward <= 0.0
    assert config.missing_reward == 0.0
    assert config.unwanted_reward == config.incorrect_reward


def event(name: str, timestamp: int) -> OutputEvent:
    return OutputEvent(Population.OUTPUT_CHAR, name, timestamp, 1.0)


def outcomes(report: EventStreamReport) -> list[EventOutcome]:
    return [record.outcome for record in report.records]


def test_correct_and_incorrect_events_are_accounted_independently() -> None:
    report = evaluate_event_stream(("m", "a"), (event("b", 0), event("m", 1), event("a", 2)))

    assert report.counts[EventOutcome.INCORRECT_EVENT] == 1
    assert report.counts[EventOutcome.CORRECT_EVENT] == 2
    assert report.event_precision == 2 / 3
    assert report.event_recall == 1.0


def test_missing_output_is_only_declared_after_patience() -> None:
    report = evaluate_event_stream(
        (TargetEvent("m", 0),),
        (),
        config=EventStreamConfig(patience_window=2),
        end_time=2,
    )
    assert outcomes(report) == [EventOutcome.MISSING_OUTPUT, EventOutcome.VALID_SILENCE]
    assert report.total_reward == -1.0


def test_silence_is_neutral_and_unwanted_output_is_penalized() -> None:
    silence = evaluate_event_stream(("m",), (event("m", 4),), end_time=5)
    unwanted = evaluate_event_stream(
        ("m",), (event("a", 1),), silence_intervals=(SilenceInterval(0, 2),)
    )

    assert silence.counts[EventOutcome.VALID_SILENCE] == 1
    assert silence.total_reward == 1.0
    assert unwanted.counts[EventOutcome.UNWANTED_OUTPUT] == 1


def test_eos_accounting_distinguishes_premature_missing_and_post_eos() -> None:
    premature = evaluate_event_stream(("m", "<EOS>"), (event("<EOS>", 0),))
    missing = evaluate_event_stream(("m", "<EOS>"), (event("m", 0),), end_time=8)
    post = evaluate_event_stream(("m", "<EOS>"), (event("m", 0), event("<EOS>", 1), event("a", 2)))

    assert premature.counts[EventOutcome.PREMATURE_EOS] == 1
    assert missing.counts[EventOutcome.MISSING_EOS] == 1
    assert post.counts[EventOutcome.POST_EOS_OUTPUT] == 1
    assert post.eos_accuracy


def test_withheld_prefix_allows_eos_after_unavailable_outputs() -> None:
    report = evaluate_event_stream(
        ("m", "a", "<EOS>"),
        (event("<EOS>", 0),),
        withheld_prefix=2,
    )

    assert report.counts[EventOutcome.MISSING_OUTPUT] == 2
    assert report.counts[EventOutcome.PREMATURE_EOS] == 0
    assert report.eos_accuracy


def test_early_timing_is_open_by_default_but_can_be_configured() -> None:
    target = (TargetEvent("m", 10),)
    tolerant = evaluate_event_stream(target, (event("m", 0),))
    strict = evaluate_event_stream(
        target, (event("m", 0),), config=EventStreamConfig(early_tolerance=2)
    )

    assert tolerant.counts[EventOutcome.CORRECT_EVENT] == 1
    assert strict.counts[EventOutcome.INCORRECT_EVENT] == 1


def test_configured_event_rewards_are_consumed_by_plasticity_and_ledger() -> None:
    agent = ConversationAgent(AgentConfig(seed=12))
    agent.runtime.plasticity.eligibility.fill(1.0)
    config = EventStreamConfig(
        correct_reward=2.0,
        incorrect_reward=-2.0,
        missing_reward=-3.0,
        unwanted_reward=-4.0,
        premature_eos_reward=-5.0,
        missing_eos_reward=-6.0,
    )
    report = evaluate_event_stream(
        ("m", "<EOS>"),
        (event("b", 0), event("<EOS>", 1)),
        config=config,
    )
    before = agent.runtime.network.synapses.weight.copy()

    prediction_error = agent.apply_event_stream_reward(report)

    assert prediction_error == report.total_reward
    assert agent.reward_ledger == report.records
    assert not np.array_equal(agent.runtime.network.synapses.weight, before)


def test_event_reward_outcomes_remain_distinct_in_the_consumed_ledger() -> None:
    agent = ConversationAgent(AgentConfig(seed=13))
    config = EventStreamConfig(patience_window=1)
    reports = (
        evaluate_event_stream(("m",), (event("m", 0),), config=config),
        evaluate_event_stream(("m",), (event("b", 0),), config=config),
        evaluate_event_stream(("m",), (), config=config, end_time=2),
        evaluate_event_stream(
            ("m",), (event("b", 0),), silence_intervals=(SilenceInterval(0, 0),), config=config
        ),
        evaluate_event_stream(("m", "<EOS>"), (event("<EOS>", 0),), config=config),
        evaluate_event_stream(("m", "<EOS>"), (event("m", 0),), config=config, end_time=8),
    )

    for report in reports:
        agent.apply_event_stream_reward(report)

    ledger_outcomes = {record.outcome for record in agent.reward_ledger}
    assert {
        EventOutcome.CORRECT_EVENT,
        EventOutcome.INCORRECT_EVENT,
        EventOutcome.MISSING_OUTPUT,
        EventOutcome.UNWANTED_OUTPUT,
        EventOutcome.PREMATURE_EOS,
        EventOutcome.MISSING_EOS,
    } <= ledger_outcomes


def test_event_stream_rejects_non_increasing_timestamps() -> None:
    with pytest.raises(ValueError, match="observed timestamps"):
        evaluate_event_stream(("m",), (event("m", 2), event("a", 1)))

    with pytest.raises(ValueError, match="target timestamps"):
        evaluate_event_stream((TargetEvent("m", 2), TargetEvent("a", 2)), ())
