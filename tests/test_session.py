import numpy as np
import pytest

from continual_agent.agent.session import (
    ConflictPolicy,
    ResponseSession,
    SessionExecutionSnapshot,
    SessionPolicy,
    SessionSnapshot,
    SessionState,
    SessionStateError,
    WeightConflictError,
)
from continual_agent.simulation.synapses import SynapticActivityState


def snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        neuron_voltage=np.array([0.2, 0.4]),
        neuron_refractory=np.array([0, 1]),
        synaptic_activity=np.array([0.5, 0.0]),
        plasticity_eligibility=np.array([0.1]),
    )


def test_session_enforces_response_lifecycle() -> None:
    session = ResponseSession()
    assert session.state is SessionState.IDLE

    with pytest.raises(SessionStateError):
        session.begin_response()

    session.begin(snapshot())
    session.begin_response()
    session.receive_eos()
    session.complete()
    assert session.state is SessionState.COMPLETE

    session.reset_for_next_response()
    assert session.state is SessionState.IDLE


def test_session_snapshot_isolated_and_contains_initial_synaptic_activity() -> None:
    initial = snapshot()
    session = ResponseSession(
        policy=SessionPolicy(reset_neuron_state=True, reset_synaptic_activity=True)
    )
    session.begin(initial)

    initial.neuron_voltage[0] = 99.0
    first = session.snapshot
    assert first is not None
    first.synaptic_activity[0] = 99.0
    second = session.snapshot
    assert second is not None
    assert second.neuron_voltage[0] == 0.2
    assert second.synaptic_activity[0] == 0.5
    assert session.policy.reset_neuron_state
    assert session.policy.reset_synaptic_activity


def test_object_snapshot_payloads_are_isolated() -> None:
    affect = {"valence": 0.2}
    session = ResponseSession()
    session.begin(
        SessionSnapshot(
            np.zeros(1), np.zeros(1, dtype=int), np.array([0.3]), affect=affect
        )
    )

    affect["valence"] = 1.0
    saved = session.snapshot
    assert saved is not None
    assert saved.affect == {"valence": 0.2}


def test_session_cannot_skip_eos_or_reset_early() -> None:
    session = ResponseSession()
    session.begin(snapshot())
    session.begin_response()

    with pytest.raises(SessionStateError):
        session.complete()
    with pytest.raises(SessionStateError):
        session.reset_for_next_response()

    session.abort()
    assert session.state is SessionState.IDLE


def test_parallel_execution_copies_neural_activity_and_weights() -> None:
    source = snapshot()
    weights = np.array([0.2, 0.4])
    execution = SessionExecutionSnapshot(source, weights)

    source.synaptic_activity[0] = 8.0
    weights[0] = 8.0
    execution.neural.synaptic_activity[1] = 7.0
    execution.weights[1] = 7.0

    np.testing.assert_array_equal(
        execution.initial_synaptic_activity_state.activity, [0.5, 0.0]
    )
    assert weights.tolist() == [8.0, 0.4]


def test_weight_delta_merges_only_recorded_indices() -> None:
    shared = np.array([1.0, 2.0, 3.0])
    execution = SessionExecutionSnapshot(snapshot(), shared)
    execution.record_weight_delta(np.array([1]), np.array([0.25]))

    execution.merge_into(shared)
    np.testing.assert_allclose(shared, [1.0, 2.25, 3.0])


def test_weight_conflict_policy_is_explicit() -> None:
    shared = np.array([1.0, 2.0])
    execution = SessionExecutionSnapshot(snapshot(), shared)
    execution.record_weight_delta(np.array([0]), np.array([0.5]))
    shared[0] = 1.25

    with pytest.raises(WeightConflictError):
        execution.merge_into(shared)
    assert shared[0] == 1.25

    execution.merge_into(shared, ConflictPolicy.SUM)
    assert shared[0] == 1.75


def test_initial_synaptic_activity_uses_weighted_decayed_average() -> None:
    state = SynapticActivityState.aggregate(
        [np.array([0.0]), np.array([1.0])], decay=0.5
    )
    assert state.activity[0] == pytest.approx(2 / 3)

    updated = state.update(np.array([0.0]), decay=0.5)
    assert updated.activity[0] == pytest.approx(2 / 7)


def test_execution_snapshot_can_aggregate_activity_baseline() -> None:
    execution = SessionExecutionSnapshot(snapshot(), np.array([0.2, 0.4]))

    updated = execution.aggregate_initial_synaptic_activity(
        np.array([1.0, 0.0]), decay=0.5
    )

    assert updated.activity[0] == pytest.approx(5 / 6)
    np.testing.assert_array_equal(
        execution.initial_synaptic_activity_state.activity, [5 / 6, 0.0]
    )
