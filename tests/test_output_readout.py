import numpy as np

from continual_agent.cognition.readout import (
    Action,
    ActionReadout,
    EventReadout,
    OutputArbitrationPolicy,
    OutputCandidate,
)
from continual_agent.simulation.population_layout import Population, PopulationLayout


def make_layout() -> PopulationLayout:
    input_count, hidden_count = 48, 48
    affect_count, action_count, char_count = 28, 28, 36
    affect_start = input_count + hidden_count
    action_start = affect_start + affect_count
    char_start = action_start + action_count
    return PopulationLayout(
        input_count=input_count, hidden_count=hidden_count,
        affect_count=affect_count, action_count=action_count, char_count=char_count,
        affect_subgroups={name: slice(affect_start + i * 4, affect_start + (i + 1) * 4)
                          for i, name in enumerate(("valence", "arousal", "uncertainty", "curiosity", "threat", "competence", "social_affiliation"))},
        action_subgroups={name: slice(action_start + i * 4, action_start + (i + 1) * 4)
                         for i, name in enumerate(("answer", "clarify", "acknowledge", "uncertain", "revise", "refuse", "wait"))},
        char_subgroups={name: slice(char_start + i * 3, char_start + (i + 1) * 3)
                       for i, name in enumerate(("<EOS>", "m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!"))},
    )


def frame() -> np.ndarray:
    return np.zeros(make_layout().total_count)


def test_action_readout_only_exposes_named_layout_groups() -> None:
    readout = ActionReadout()

    current = make_layout()
    groups = readout.groups(current)

    assert groups[Action.ANSWER].tolist() == list(
        range(*current.subgroup(Population.OUTPUT_ACTION, "answer").indices(
            current.total_count
        ))
    )
    assert not hasattr(readout, "policy_weights")


def test_repeated_character_requires_release() -> None:
    current = make_layout()
    readout = EventReadout(current, activation_threshold=1.0, release_threshold=0.5)
    character = current.subgroup(Population.OUTPUT_CHAR, "m")

    first = frame()
    first[character] = 1.0
    event = readout.observe(first, timestamp=0)
    assert event is not None and event.name == "m"
    assert readout.observe(first, timestamp=1) is None  # sustained activation

    assert readout.observe(frame(), timestamp=2) is None
    released = frame()
    released[character] = 1.0
    event = readout.observe(released, timestamp=3)
    assert event is not None
    assert event.name == "m"
    assert event.interval == 3


def test_silence_is_not_eos() -> None:
    readout = EventReadout(make_layout())
    assert readout.observe(frame()) is None
    assert not readout.stopped
    assert readout.events == []


def test_eos_is_an_event_and_suppresses_later_output() -> None:
    current = make_layout()
    readout = EventReadout(current)
    eos = current.subgroup(Population.OUTPUT_CHAR, "<EOS>")
    eos_frame = frame()
    eos_frame[eos] = 1.0

    event = readout.observe(eos_frame, timestamp=4)
    assert event is not None and event.is_eos
    assert readout.stopped

    character = current.subgroup(Population.OUTPUT_CHAR, "a")
    later = frame()
    later[character] = 1.0
    assert readout.observe(later, timestamp=5) is None
    assert len(readout.events) == 1


def test_sustained_activation_emits_once() -> None:
    current = make_layout()
    readout = EventReadout(current)
    action = current.subgroup(Population.OUTPUT_ACTION, "answer")
    active = frame()
    active[action] = 1.0

    event = readout.observe(active)
    assert event is not None and event.name == "answer"
    assert readout.observe(active) is None
    assert readout.observe(active) is None


def test_arbitration_prefers_stronger_evidence_across_characters() -> None:
    policy = OutputArbitrationPolicy()
    decision = policy.arbitrate(
        (
            OutputCandidate(Population.OUTPUT_CHAR, "m", 2.0, 0),
            OutputCandidate(Population.OUTPUT_CHAR, "a", 3.0, 1),
        )
    )
    assert decision.selected is not None and decision.selected.name == "a"
    assert decision.reason == "selected"


def test_arbitration_ties_are_explicit_for_eos_and_action() -> None:
    policy = OutputArbitrationPolicy()
    eos = OutputCandidate(Population.OUTPUT_CHAR, "<EOS>", 2.0, 0)
    character = OutputCandidate(Population.OUTPUT_CHAR, "m", 2.0, 1)
    action = OutputCandidate(Population.OUTPUT_ACTION, "answer", 2.0, 0)

    assert policy.arbitrate((character, eos)).selected == eos
    assert policy.arbitrate((character, action)).selected == action


def test_readout_emits_only_the_strongest_simultaneous_character() -> None:
    current = make_layout()
    readout = EventReadout(current)
    weaker = current.subgroup(Population.OUTPUT_CHAR, "m")
    stronger = current.subgroup(Population.OUTPUT_CHAR, "a")
    active = frame()
    active[weaker] = 1.0
    active[stronger] = 2.0

    event = readout.observe(active, timestamp=0)
    assert event is not None and event.name == "a"
    assert readout.last_arbitration is not None
    assert [candidate.name for candidate in readout.last_arbitration.candidates] == ["m", "a"]


def test_arbitration_priorities_are_configurable() -> None:
    policy = OutputArbitrationPolicy(eos_priority=-1)
    eos = OutputCandidate(Population.OUTPUT_CHAR, "<EOS>", 2.0, 0)
    character = OutputCandidate(Population.OUTPUT_CHAR, "m", 2.0, 1)
    assert policy.arbitrate((eos, character)).selected == character


def test_global_active_output_blocks_other_candidates_until_release() -> None:
    current = make_layout()
    readout = EventReadout(current, release_threshold=0.5)
    action = current.subgroup(Population.OUTPUT_ACTION, "answer")
    character = current.subgroup(Population.OUTPUT_CHAR, "m")
    first = frame()
    first[action] = 1.0
    active = frame()
    active[action] = 1.0
    active[character] = 1.0

    assert readout.observe(first, timestamp=0) is not None
    assert readout.observe(active, timestamp=1) is None
    assert readout.last_arbitration is not None
    assert readout.last_arbitration.reason == "active"
    assert readout.last_arbitration.selected is None
    released = frame()
    released[character] = 1.0
    assert readout.observe(released, timestamp=2) is not None
    assert readout.events[-1].name == "m"
    assert readout.events[-1].interval == 2


def test_thresholds_require_hysteresis_order() -> None:
    with np.testing.assert_raises(ValueError):
        EventReadout(make_layout(), activation_threshold=1.0, release_threshold=1.0)
