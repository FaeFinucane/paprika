import pytest

from continual_agent.cognition.affect import AffectiveEvent, AffectiveState
from continual_agent.cognition.readout import Action
from continual_agent.environment.scenarios import ConversationScenario
from continual_agent.evaluation.event_stream import EventStreamConfig, TargetEvent

# Not that useful as a test
def test_affect_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        AffectiveState(valence=float("nan"))
    with pytest.raises(ValueError):
        AffectiveState().observe(AffectiveEvent(novelty=float("inf")))


def test_scenario_validates_targets_and_messages() -> None:
    with pytest.raises(ValueError):
        ConversationScenario("", ("hello",), Action.ANSWER)
    with pytest.raises(ValueError):
        ConversationScenario(
            "test", ("hello",), Action.ANSWER, affect_targets={"valence": (1.0, 0.0)}
        )


def test_event_boundaries_validate_inputs() -> None:
    with pytest.raises(ValueError):
        TargetEvent("answer", -1)
    with pytest.raises(ValueError):
        TargetEvent("")
    with pytest.raises(ValueError):
        EventStreamConfig(patience_window=-1)
