"""End-to-end contracts for the named output event stream."""

from continual_agent.agent.conversation_agent import ConversationAgent, NetworkConfig
from continual_agent.cognition.readout import OutputEvent
from continual_agent.evaluation.event_stream import (
    EventOutcome,
    EventStreamConfig,
    TargetEvent,
    evaluate_event_stream,
)
from continual_agent.simulation.population_layout import Population


def output(name: str, timestamp: int) -> OutputEvent:
    return OutputEvent(Population.OUTPUT_CHAR, name, timestamp, 1.0)


def test_named_sequence_accepts_late_events_in_order_and_reports_latency() -> None:
    report = evaluate_event_stream(
        (TargetEvent("red", 2), TargetEvent("green", 5), TargetEvent("<EOS>", 7)),
        (output("red", 3), output("green", 6), output("<EOS>", 8)),
        config=EventStreamConfig(patience_window=1, measure_timing_error=True),
        end_time=8,
    )

    assert [record.outcome for record in report.records[:3]] == [
        EventOutcome.CORRECT_EVENT,
        EventOutcome.CORRECT_EVENT,
        EventOutcome.CORRECT_EVENT,
    ]
    assert report.latencies == [1, 1, 1]
    assert report.timing_errors == [1, 1, 1]
    assert report.eos_accuracy


def test_agent_consumes_stream_report_as_one_ordered_reward_ledger() -> None:
    agent = ConversationAgent(NetworkConfig(seed=31))
    report = evaluate_event_stream(("x", "<EOS>"), (output("x", 0), output("<EOS>", 1)))

    prediction_error = agent.apply_event_stream_reward(report)

    assert prediction_error == report.total_reward
    assert agent.reward_ledger == report.records
    assert [record.label for record in agent.reward_ledger] == ["x", "<EOS>"]
