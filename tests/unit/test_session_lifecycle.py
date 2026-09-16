import numpy as np
import pytest
from src.interaction.plugins import Drives
from src.network.adjustments import NetworkAdjustment
from src.network.definition import NetworkDefinition
from src.network.population import NeuronPopulationSpec
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
    snn = NetworkDefinition((NeuronPopulationSpec("POPULATION", 1),), ()).compile(
        np.random.default_rng(1)
    )
    events: list[str] = []
    session = Session.build(
        snn,
        [_Drive(events), _Adaptation(snn, events, 0.1), _Adaptation(snn, events, 0.2)],
    )

    session.tick()

    assert events == ["produce", "observe:0.1", "observe:0.2", "propose:0.1", "propose:0.2"]


def test_session_can_freeze_all_adaptation_lifecycle_steps():
    snn = NetworkDefinition((NeuronPopulationSpec("POPULATION", 1),), ()).compile(
        np.random.default_rng(1)
    )
    events: list[str] = []
    session = Session.build(snn, [_Drive(events), _Adaptation(snn, events, 0.1)])

    with session.frozen_adaptations():
        session.tick()

    assert events == ["produce"]
    session.tick()
    assert events == ["produce", "produce", "observe:0.1", "propose:0.1"]
