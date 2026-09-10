import pytest
from src.interaction.td import TDComparison, TemporalDifferenceComparator

pytestmark = pytest.mark.unit


def test_exact_formula_and_single_advance():
    comparator = TemporalDifferenceComparator(previous_positive=1.0, previous_negative=2.0)
    result = comparator.evaluate_transition(3.0, 4.0, 5.0, 6.0, 0.5)
    assert result == TDComparison(4.5, 5.0, -0.5)
    assert comparator.previous_positive == 5.0
    assert comparator.previous_negative == 6.0


def test_positive_and_negative_rewards_are_separate_unipolar_terms():
    result = TemporalDifferenceComparator().evaluate_transition(2.0, 3.0, 0.0, 0.0, 0.9)
    assert result.positive_term == 2.0
    assert result.negative_term == 3.0
    assert result.delta == -1.0


def test_required_arguments_and_invalid_gamma():
    with pytest.raises(TypeError):
        TemporalDifferenceComparator().evaluate_transition(1, 2, 3, 4)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        TemporalDifferenceComparator().evaluate_transition(0, 0, 0, 0, 1.1)
    with pytest.raises(ValueError):
        TemporalDifferenceComparator().evaluate_transition(0, 0, 0, 0, float("nan"))


def test_reset_restores_baseline_and_allows_id_reuse():
    comparator = TemporalDifferenceComparator()
    comparator.evaluate_transition(1, 1, 4, 5, 0.5, transition_id="a")
    comparator.reset()
    result = comparator.evaluate_transition(0, 0, 2, 3, 0.5, transition_id="a")
    assert result == TDComparison(1.0, 1.5, -0.5)


def test_duplicate_transition_id_is_rejected_without_advancing():
    comparator = TemporalDifferenceComparator()
    comparator.evaluate_transition(1, 2, 3, 4, 0.5, transition_id=7)
    with pytest.raises(ValueError, match="already evaluated"):
        comparator.evaluate_transition(9, 9, 10, 10, 0.5, transition_id=7)
    assert (comparator.previous_positive, comparator.previous_negative) == (3.0, 4.0)
