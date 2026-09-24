"""Contract for declarative network construction."""

import pytest

from src.builder import NetworkBuilder
from src.diagnostics import inspect_network
from src.interaction.drives import TonicDrive, TonicDriveSpec
from src.network.connectivity import FanInSpec, FanOutSpec, StrengthSpec

pytestmark = pytest.mark.unit


def test_network_builder_compiles_a_reusable_basic_session():
    builder = NetworkBuilder()
    builder.add_population("SOURCE", 2, plugins=(TonicDriveSpec(0.2),))
    builder.add_population("TARGET", 2)
    builder.connect("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2))
    session = builder.compile(1)
    snn = session.snn
    assert snn.layout.population("SOURCE").count == 2
    assert any(isinstance(hook, TonicDrive) for hook in session.drive_sources)
    assert inspect_network(snn).valid


def test_connection_can_be_retrieved_and_edited_through_population_handles():
    builder = NetworkBuilder()
    source = builder.add_population("SOURCE", 4)
    target = builder.add_population("TARGET", 4)
    declared = builder.connect(source, target, FanOutSpec(1), StrengthSpec(0.2))

    connection = builder.connection(source, target)
    assert connection is declared
    connection.topology = FanInSpec(2)
    connection.strength.mean = 0.3

    session = builder.compile(1)
    mask = session.snn.synapses.projection_mask("SOURCE_to_TARGET")
    assert mask.any()
    assert session.snn.synapses.strength[mask].mean() == 0.3


def test_direct_plugin_builds_hooks_when_the_builder_compiles():
    installed = []

    class DirectPlugin:
        def build_hooks(self, session, _rng):
            installed.append(session)
            return ()

    builder = NetworkBuilder()
    builder.add_population("POPULATION", 1)
    builder.add_plugin(DirectPlugin())

    session = builder.compile(1)

    assert installed == [session]
