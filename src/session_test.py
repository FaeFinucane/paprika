"""Contract for session hook lifecycle ordering."""

import pytest

from src.builder import NetworkBuilder
from src.interaction.plugins import Drives
from src.network.adjustments import NetworkAdjustment
from src.network.snn import SNN, Spikes
from src.session import Session

pytestmark = pytest.mark.unit


class _Drive:
    def __init__(self, events: list[str]):
        self.events = events

    def produce(self) -> Drives:
        self.events.append("produce")
        return Drives()


class _Adaptation:
    def __init__(self, snn: SNN, events: list[str], delta: float):
        self.snn = snn
        self.events = events
        self.delta = delta

    def observe(self, _spikes: Spikes) -> None:
        self.events.append(f"observe:{self.delta}")

    def propose(self, _snn: SNN) -> NetworkAdjustment:
        self.events.append(f"propose:{self.delta}")
        return NetworkAdjustment()


def test_session_observes_then_commits_all_adaptations_from_one_registration_list():
    builder = NetworkBuilder()
    builder.add_population("POPULATION", 1)
    snn = builder.compile(1).snn
    events: list[str] = []
    session = Session.build(
        snn,
        [_Drive(events), _Adaptation(snn, events, 0.1), _Adaptation(snn, events, 0.2)],
    )

    session.tick()

    assert events == ["produce", "observe:0.1", "observe:0.2", "propose:0.1", "propose:0.2"]


def test_session_can_freeze_all_adaptation_lifecycle_steps():
    builder = NetworkBuilder()
    builder.add_population("POPULATION", 1)
    snn = builder.compile(1).snn
    events: list[str] = []
    session = Session.build(snn, [_Drive(events), _Adaptation(snn, events, 0.1)])

    with session.frozen_adaptations():
        session.tick()

    assert events == ["produce"]
    session.tick()
    assert events == ["produce", "produce", "observe:0.1", "propose:0.1"]


def test_session_add_registers_every_lifecycle_role_once():
    builder = NetworkBuilder()
    builder.add_population("POPULATION", 1)
    session = builder.compile(1)
    events: list[str] = []
    adaptation = _Adaptation(session.snn, events, 0.1)

    session.add(adaptation)

    assert adaptation in session.observers
    assert adaptation in session.adaptations
    with pytest.raises(ValueError, match="already registered"):
        session.add(adaptation)

    session.tick()
    assert events == ["observe:0.1", "propose:0.1"]


def test_session_remove_unregisters_every_lifecycle_role_and_can_be_reversed():
    builder = NetworkBuilder()
    builder.add_population("POPULATION", 1)
    session = builder.compile(1)
    events: list[str] = []
    adaptation = _Adaptation(session.snn, events, 0.1)
    session.add(adaptation)

    session.remove(adaptation)

    assert all(observer is not adaptation for observer in session.observers)
    assert all(registered is not adaptation for registered in session.adaptations)
    session.tick()
    assert events == []
    with pytest.raises(ValueError, match="not registered"):
        session.remove(adaptation)

    session.add(adaptation)
    session.tick()
    assert events == ["observe:0.1", "propose:0.1"]
