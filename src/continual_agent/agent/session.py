"""Explicit lifecycle and state policy for one response session.

This module deliberately does not own a network.  A session owns its lifecycle
metadata and a copied starting snapshot, so callers cannot accidentally use a
previous session's mutable arrays as the next session's baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

import numpy as np

from continual_agent.simulation.synapses import SynapticActivityState


class SessionState(str, Enum):
    IDLE = "idle"
    RECEIVING_INPUT = "receiving_input"
    RESPONDING = "responding"
    EOS_RECEIVED = "eos_received"
    COMPLETE = "complete"
    EXHAUSTED = "exhausted"


class SessionStateError(RuntimeError):
    """Raised when a lifecycle operation is invalid for the current state."""


class WeightConflictError(ValueError):
    """Raised when a session update would overwrite a newer shared weight."""


class ConflictPolicy(str, Enum):
    """How a delta is merged when its shared base value has changed.

    ``REJECT`` is the safe default. ``SUM`` applies both changes, while
    ``AVERAGE`` averages the current shared value with the session's proposed
    value. Neither policy replaces network objects or connectivity.
    """

    REJECT = "reject"
    SUM = "sum"
    AVERAGE = "average"


class InputSignal(str, Enum):
    """Control signals for an externally presented input message."""

    INPUT_BEGIN = "input_begin"
    INPUT_END = "input_end"


@dataclass(frozen=True)
class SessionPolicy:
    """Reset/persistence decisions at a response boundary.

    These defaults describe the current vertical slice: neural recurrent state,
    affect, working memory, and plasticity traces persist; readout evidence is
    response-local. The policy is data so a later network adapter can apply it
    without changing lifecycle semantics.
    """

    reset_readout: bool = True
    reset_neuron_state: bool = False
    reset_synaptic_activity: bool = False
    persist_affect: bool = True
    persist_working_memory: bool = True
    persist_plasticity_eligibility: bool = True


@dataclass
class SessionSnapshot:
    """Starting state for a session, copied at construction and on access."""

    neuron_voltage: np.ndarray
    neuron_refractory: np.ndarray
    synaptic_activity: np.ndarray
    affect: object | None = None
    working_memory: object | None = None
    plasticity_eligibility: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        self.neuron_voltage = np.array(self.neuron_voltage, dtype=float, copy=True)
        self.neuron_refractory = np.array(self.neuron_refractory, dtype=np.int64, copy=True)
        self.synaptic_activity = np.array(self.synaptic_activity, dtype=float, copy=True)
        self.plasticity_eligibility = np.array(
            self.plasticity_eligibility, dtype=float, copy=True
        )
        if self.neuron_voltage.shape != self.neuron_refractory.shape:
            raise ValueError("neuron voltage and refractory state must have equal shapes")

    def copy(self) -> "SessionSnapshot":
        return SessionSnapshot(
            self.neuron_voltage,
            self.neuron_refractory,
            self.synaptic_activity,
            deepcopy(self.affect),
            deepcopy(self.working_memory),
            self.plasticity_eligibility,
        )


@dataclass(frozen=True)
class WeightDelta:
    """A validated sparse update against a particular weight baseline."""

    indices: np.ndarray
    delta: np.ndarray
    base: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        delta = np.asarray(self.delta, dtype=float)
        base = np.asarray(self.base, dtype=float)
        if indices.ndim != 1 or delta.shape != indices.shape or base.shape != indices.shape:
            raise ValueError("weight delta indices, delta, and base must have equal 1D shapes")
        if np.any(indices < 0) or np.unique(indices).size != indices.size:
            raise ValueError("weight delta indices must be unique and non-negative")
        if not np.isfinite(delta).all() or not np.isfinite(base).all():
            raise ValueError("weight delta values must be finite")
        object.__setattr__(self, "indices", indices.copy())
        object.__setattr__(self, "delta", delta.copy())
        object.__setattr__(self, "base", base.copy())


@dataclass
class SessionExecutionSnapshot:
    """Independent execution state: copied neural/activity state and weights."""

    neural: SessionSnapshot
    weights: np.ndarray
    _base_weights: np.ndarray = field(init=False, repr=False)
    _initial_synaptic_state: SynapticActivityState = field(init=False, repr=False)
    _updates: list[WeightDelta] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.neural = self.neural.copy()
        self._initial_synaptic_state = SynapticActivityState(self.neural.synaptic_activity)
        self.weights = np.asarray(self.weights, dtype=float).copy()
        if self.weights.ndim != 1 or not np.isfinite(self.weights).all():
            raise ValueError("session weights must be a finite 1D array")
        self._base_weights = self.weights.copy()

    @property
    def initial_synaptic_activity_state(self) -> SynapticActivityState:
        """Activity baseline with an aggregation-ready weighted sample count."""

        return self._initial_synaptic_state.copy()

    def aggregate_initial_synaptic_activity(
        self,
        observation: np.ndarray,
        *,
        weight: float = 1.0,
        decay: float = 0.9,
    ) -> SynapticActivityState:
        """Update the short-term baseline without coupling it to long-term weights."""

        self._initial_synaptic_state = self._initial_synaptic_state.update(
            observation, weight=weight, decay=decay
        )
        return self._initial_synaptic_state.copy()

    @property
    def updates(self) -> tuple[WeightDelta, ...]:
        return tuple(self._updates)

    def record_weight_delta(self, indices: np.ndarray, delta: np.ndarray) -> None:
        """Record and apply a sparse local update without exposing shared weights."""

        indices = np.asarray(indices, dtype=np.int64)
        delta = np.asarray(delta, dtype=float)
        if indices.ndim != 1 or delta.shape != indices.shape:
            raise ValueError("indices and delta must have equal 1D shapes")
        if np.any(indices < 0) or np.any(indices >= self.weights.size):
            raise ValueError("weight delta index is out of bounds")
        if not np.isfinite(delta).all():
            raise ValueError("weight delta values must be finite")
        self._updates.append(WeightDelta(indices, delta, self.weights[indices]))
        self.weights[indices] += delta

    def record_weight_update(self, indices: np.ndarray, updated: np.ndarray) -> None:
        """Record absolute local values as deltas against the local baseline."""

        indices = np.asarray(indices, dtype=np.int64)
        updated = np.asarray(updated, dtype=float)
        if indices.ndim != 1 or updated.shape != indices.shape:
            raise ValueError("indices and updated values must have equal 1D shapes")
        if np.any(indices < 0) or np.any(indices >= self.weights.size):
            raise ValueError("weight update index is out of bounds")
        self.record_weight_delta(indices, updated - self.weights[indices])

    def merge_into(
        self,
        shared_weights: np.ndarray,
        policy: ConflictPolicy = ConflictPolicy.REJECT,
    ) -> None:
        """Merge recorded deltas into shared weights under an explicit policy."""

        shared = np.asarray(shared_weights, dtype=float)
        if shared.ndim != 1 or shared.shape != self.weights.shape:
            raise ValueError("shared weights must have the same 1D shape as the session")
        try:
            policy = ConflictPolicy(policy)
        except ValueError as exc:
            raise ValueError(f"unknown weight conflict policy: {policy}") from exc
        if policy is ConflictPolicy.REJECT:
            # Preflight the complete batch so rejection is atomic.
            for update in self._updates:
                if np.any(
                    ~np.isclose(
                        shared[update.indices], update.base, rtol=0.0, atol=0.0
                    )
                ):
                    raise WeightConflictError(
                        "shared weights changed under a session update"
                    )
        for update in self._updates:
            current = shared[update.indices]
            proposed = update.base + update.delta
            if policy is ConflictPolicy.SUM:
                shared[update.indices] = current + update.delta
            elif policy is ConflictPolicy.AVERAGE:
                shared[update.indices] = (current + proposed) / 2.0
            else:
                shared[update.indices] = proposed


@dataclass
class ResponseSession:
    """Lifecycle boundary for one externally initiated response."""

    policy: SessionPolicy = field(default_factory=SessionPolicy)
    state: SessionState = SessionState.IDLE
    _snapshot: SessionSnapshot | None = field(default=None, repr=False)
    input_active: bool = False

    @property
    def snapshot(self) -> SessionSnapshot | None:
        return self._snapshot.copy() if self._snapshot is not None else None

    def begin(self, snapshot: SessionSnapshot) -> None:
        self._require(SessionState.IDLE)
        self._snapshot = snapshot.copy()
        self.state = SessionState.RECEIVING_INPUT

    def begin_response(self) -> None:
        self._require(SessionState.RECEIVING_INPUT)
        if self.input_active:
            raise SessionStateError("cannot begin response while input is active")
        self.state = SessionState.RESPONDING

    def begin_input(self) -> None:
        """Open one input presentation window; overlapping input is rejected."""

        self._require(SessionState.RECEIVING_INPUT)
        if self.input_active:
            raise SessionStateError("input presentation is already active")
        self.input_active = True

    def end_input(self) -> None:
        self._require(SessionState.RECEIVING_INPUT)
        if not self.input_active:
            raise SessionStateError("input presentation is not active")
        self.input_active = False

    def accept_input_frame(self) -> None:
        """Assert that a semantic frame is inside the input boundary."""

        self._require(SessionState.RECEIVING_INPUT)
        if not self.input_active:
            raise SessionStateError("input frame is outside an active presentation")

    def handle_input_signal(self, signal: InputSignal) -> None:
        if signal is InputSignal.INPUT_BEGIN:
            self.begin_input()
        elif signal is InputSignal.INPUT_END:
            self.end_input()
        else:
            raise ValueError(f"unknown input signal: {signal}")

    def receive_eos(self) -> None:
        self._require(SessionState.RESPONDING)
        self.state = SessionState.EOS_RECEIVED

    def complete(self, snapshot: SessionSnapshot | None = None) -> None:
        self._require(SessionState.EOS_RECEIVED)
        if snapshot is not None:
            self._snapshot = snapshot.copy()
        self.state = SessionState.COMPLETE

    def reset_for_next_response(self, snapshot: SessionSnapshot | None = None) -> None:
        if self.state not in (
            SessionState.EOS_RECEIVED,
            SessionState.COMPLETE,
            SessionState.EXHAUSTED,
        ):
            raise SessionStateError(
                f"cannot reset for next response from {self.state.value}"
            )
        self._snapshot = snapshot.copy() if snapshot is not None else self._snapshot
        self.input_active = False
        self.state = SessionState.IDLE

    def abort(self, snapshot: SessionSnapshot | None = None) -> None:
        """Discard an unfinished response without treating it as EOS."""

        if self.state is not SessionState.RESPONDING:
            raise SessionStateError(f"cannot abort from {self.state.value}")
        if snapshot is not None:
            self._snapshot = snapshot.copy()
        self.state = SessionState.IDLE
        self.input_active = False

    def exhaust(self, snapshot: SessionSnapshot | None = None) -> None:
        """Record that response output ended at its configured limit."""

        self._require(SessionState.RESPONDING)
        if snapshot is not None:
            self._snapshot = snapshot.copy()
        self.state = SessionState.EXHAUSTED
        self.input_active = False

    def _require(self, expected: SessionState) -> None:
        if self.state is not expected:
            raise SessionStateError(
                f"expected {expected.value}, current state is {self.state.value}"
            )
