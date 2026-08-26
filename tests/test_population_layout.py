import pytest

from continual_agent.simulation.population_layout import Population, PopulationLayout


def make_layout() -> PopulationLayout:
    input_count, hidden_count = 48, 48
    affect_count, action_count, char_count = 28, 28, 36
    affect_start = input_count + hidden_count
    action_start = affect_start + affect_count
    char_start = action_start + action_count
    return PopulationLayout(
        input_count=input_count,
        hidden_count=hidden_count,
        affect_count=affect_count,
        action_count=action_count,
        char_count=char_count,
        affect_subgroups={
            name: slice(affect_start + index * 4, affect_start + (index + 1) * 4)
            for index, name in enumerate(
                (
                    "valence",
                    "arousal",
                    "uncertainty",
                    "curiosity",
                    "threat",
                    "competence",
                    "social_affiliation",
                )
            )
        },
        action_subgroups={
            name: slice(action_start + index * 4, action_start + (index + 1) * 4)
            for index, name in enumerate(
                ("answer", "clarify", "acknowledge", "uncertain", "revise", "refuse", "wait")
            )
        },
        char_subgroups={
            name: slice(char_start + index * 3, char_start + (index + 1) * 3)
            for index, name in enumerate(
                ("<EOS>", "m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!")
            )
        },
    )


def test_default_populations_are_disjoint_and_sized():
    current = make_layout()
    ranges = [range(current.slice(pop).start, current.slice(pop).stop) for pop in Population]
    assert sum(len(items) for items in ranges) == current.total_count == 188
    assert len(set().union(*map(set, ranges))) == current.total_count


def test_default_subgroups_have_expected_bounds_and_sizes():
    current = make_layout()
    for population, groups, size in (
        (Population.AFFECT, current.affect_subgroups, 4),
        (Population.OUTPUT_ACTION, current.action_subgroups, 4),
        (Population.OUTPUT_CHAR, current.char_subgroups, 3),
    ):
        bounds = current.slice(population)
        assert all(
            group.start >= bounds.start
            and group.stop <= bounds.stop
            and group.stop - group.start == size
            for group in groups.values()
        )
    assert "<EOS>" in current.char_subgroups


def test_subgroup_selection_is_semantic():
    current = make_layout()
    assert current.subgroup(Population.AFFECT, "valence") == slice(96, 100)
    assert current.subgroup(Population.OUTPUT_ACTION, "answer") == slice(124, 128)
    assert current.subgroup(Population.OUTPUT_CHAR, "<EOS>") == slice(152, 155)


def test_invalid_subgroup_bounds_are_rejected():
    with pytest.raises(ValueError):
        PopulationLayout(affect_subgroups={"bad": slice(0, 2)})
    with pytest.raises(ValueError):
        PopulationLayout(char_subgroups={"bad": slice(1, 3, 2)})


def test_subgroups_must_belong_to_their_named_population():
    with pytest.raises(ValueError, match="out-of-bounds"):
        PopulationLayout(affect_subgroups={"bad": slice(0, 2)})


def test_named_subgroup_lookup_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown output_char subgroup"):
        make_layout().subgroup(Population.OUTPUT_CHAR, "missing")
