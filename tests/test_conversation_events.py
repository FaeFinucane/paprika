from dataclasses import replace

import numpy as np
import pytest

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.agent.session import InputSignal, SessionState, SessionStateError
from continual_agent.cognition.readout import Action
from continual_agent.simulation.population_layout import Population


def test_baseline_current_does_not_activate_candidate_outputs() -> None:
    agent = ConversationAgent()

    current = agent._frame_current(np.zeros(agent.config.input_features))

    np.testing.assert_array_equal(
        current[agent.runtime.layout.slice(Population.OUTPUT_ACTION)], 0.0
    )
    np.testing.assert_array_equal(current[agent.runtime.layout.slice(Population.OUTPUT_CHAR)], 0.0)
    np.testing.assert_array_equal(current[agent.runtime.layout.slice(Population.AFFECT)], 1.01)


def test_nonblank_current_does_not_tonic_drive_action_outputs() -> None:
    agent = ConversationAgent()

    current = agent._frame_current(np.ones(agent.config.input_features))

    np.testing.assert_array_equal(
        current[agent.runtime.layout.slice(Population.OUTPUT_ACTION)], 0.0
    )
    np.testing.assert_array_equal(current[agent.runtime.layout.slice(Population.OUTPUT_CHAR)], 0.0)


def test_untrained_response_does_not_emit_baseline_eos_or_character() -> None:
    agent = ConversationAgent()

    response = agent.generate_response(Action.ANSWER, max_tokens=1)

    assert response.tokens == ()
    assert not response.stopped_on_eos
    assert agent.runtime.output_readout.events == []


def test_generate_response_emits_repeats_and_eos_only_as_readout_events() -> None:
    agent = ConversationAgent()
    zero = np.zeros(agent.runtime.layout.total_count)

    def output(name: str, amount: float = 1.0) -> np.ndarray:
        frame = zero.copy()
        frame[agent.runtime.layout.subgroup(Population.OUTPUT_CHAR, name)] = amount
        return frame

    frames = iter((output("m"), zero, output("m"), output("<EOS>")))
    agent.runtime.network.step = lambda current: next(frames)

    response = agent.generate_response(Action.ANSWER, max_tokens=4)

    assert response.tokens == ("m", "m")
    assert response.text == "mm"
    assert response.stopped_on_eos
    assert [event.name for event in agent.runtime.output_readout.events] == ["m", "m", "<EOS>"]


def test_generate_response_silence_does_not_become_character_or_eos() -> None:
    agent = ConversationAgent()
    zero = np.zeros(agent.runtime.layout.total_count)
    agent.config = replace(agent.config, max_response_ticks=3)
    agent.runtime.network.step = lambda current: zero.copy()

    response = agent.generate_response(Action.ANSWER, max_tokens=2)

    assert response.tokens == ()
    assert not response.stopped_on_eos
    assert agent.runtime.output_readout.events == []


def test_host_action_response_uses_event_arbitration() -> None:
    agent = ConversationAgent()
    zero = np.zeros(agent.runtime.layout.total_count)
    frame = zero.copy()
    answer = agent.runtime.layout.subgroup(Population.OUTPUT_ACTION, "answer")
    clarify = agent.runtime.layout.subgroup(Population.OUTPUT_ACTION, "clarify")
    frame[answer.start] = 1.0
    frame[clarify.start : clarify.start + 2] = 1.0
    agent.runtime.network.step = lambda current: frame.copy()

    response = agent.respond("")

    assert response.action is Action.CLARIFY
    assert response.evidence[Action.CLARIFY] > response.evidence[Action.ANSWER]
    assert agent.runtime.output_readout.last_arbitration is not None
    assert agent.runtime.output_readout.last_arbitration.selected is not None


def test_failed_raw_input_stream_cleans_up_runtime() -> None:
    agent = ConversationAgent()

    with pytest.raises(SessionStateError):
        agent.run_input_events(
            (
                InputSignal.INPUT_BEGIN,
                np.zeros(agent.config.input_features),
                InputSignal.INPUT_BEGIN,
            ),
            response_ticks=1,
        )

    assert agent.runtime.response_session.state is SessionState.IDLE
    assert not agent.runtime.response_session.input_active
    assert agent.runtime.output_readout.events == []
