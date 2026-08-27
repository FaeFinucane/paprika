import pytest

from continual_agent.evaluation.event_stream import EventStreamConfig, TargetEvent


# Not that useful as a test
def test_event_boundaries_validate_inputs() -> None:
    with pytest.raises(ValueError):
        TargetEvent("answer", -1)
    with pytest.raises(ValueError):
        TargetEvent("")
    with pytest.raises(ValueError):
        EventStreamConfig(patience_window=-1)
