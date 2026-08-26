import pytest

from continual_agent.simulation.population_layout import DEFAULT_LAYOUT, Population, PopulationLayout


def test_default_populations_are_disjoint_and_sized():
    layout = DEFAULT_LAYOUT
    ranges = [range(layout.slice(pop).start, layout.slice(pop).stop) for pop in Population]
    assert sum(len(items) for items in ranges) == layout.total_count == 188
    assert len(set().union(*map(set, ranges))) == layout.total_count


def test_default_subgroups_have_expected_bounds_and_sizes():
    layout = DEFAULT_LAYOUT
    for population, groups, size in (
        (Population.AFFECT, layout.affect_subgroups, 4),
        (Population.OUTPUT_ACTION, layout.action_subgroups, 4),
        (Population.OUTPUT_CHAR, layout.char_subgroups, 3),
    ):
        bounds = layout.slice(population)
        assert all(group.start >= bounds.start and group.stop <= bounds.stop and group.stop - group.start == size for group in groups.values())
    assert "<EOS>" in layout.char_subgroups


def test_subgroup_selection_is_semantic():
    assert DEFAULT_LAYOUT.subgroup(Population.AFFECT, "valence") == slice(96, 100)
    assert DEFAULT_LAYOUT.subgroup(Population.OUTPUT_ACTION, "answer") == slice(124, 128)
    assert DEFAULT_LAYOUT.subgroup(Population.OUTPUT_CHAR, "<EOS>") == slice(152, 155)


def test_invalid_subgroup_bounds_are_rejected():
    with pytest.raises(ValueError):
        PopulationLayout(affect_subgroups={"bad": slice(0, 2)})
    with pytest.raises(ValueError):
        PopulationLayout(char_subgroups={"bad": slice(1, 3, 2)})
