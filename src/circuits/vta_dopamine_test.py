"""Viability envelope for the reusable VTA dopamine motif."""

import numpy as np
import pytest
from tests.support.perturbations import set_membrane_voltage

from src.builder import NetworkBuilder
from src.diagnostics import inspect_network
from src.interaction.rates import UnipolarRateInput
from src.network.connectivity import FanInSpec, StrengthSpec

from .vta_dopamine import add_vta_dopamine, compiled_vta_dopamine

pytestmark = pytest.mark.viability


def test_vta_core_has_tonic_da_and_accepts_parent_declared_signed_outcomes():
    for seed in range(4):
        session, populations, vta, positive, negative = _build_vta(seed)
        _tick(session, 240)
        baseline = _modulation(session, vta, 60)
        assert abs(float(np.mean(baseline))) <= 0.02

        positive.write(0.7, 10)
        positive_response = _modulation(session, vta, 30)
        negative.write(0.7, 10)
        negative_response = _modulation(session, vta, 30)

        assert max(positive_response) >= 0.20
        assert min(negative_response) <= -0.15
        assert inspect_network(session.snn).valid
        assert populations[vta.dopamine_population.spec.name] == vta.dopamine_population


def test_vta_core_owns_only_its_intrinsic_inhibitory_projection():
    builder = NetworkBuilder()
    vta_handle = add_vta_dopamine(builder)

    assert tuple(connection.projection_name for connection in builder.connections) == (
        "VTA_INHIB_to_VTA_DA",
    )
    projection = builder.connection(vta_handle.inhibitory, vta_handle.dopamine)
    assert projection.learning == "dopamine_stdp"
    assert not projection.scalable


def test_vta_inhibition_suppresses_da_through_an_ordinary_learnable_projection():
    session, _, vta, _, _ = _build_vta(2)
    _tick(session, 240)
    synapses = session.snn.synapses
    mask = synapses.projection_mask("VTA_INHIB_to_VTA_DA")
    assert mask.any()
    assert set(synapses.learning[mask]) == {"dopamine_stdp"}

    set_membrane_voltage(session.snn, vta.inhibitory, voltage=3.0)
    assert min(_modulation(session, vta, 20)) <= -0.06


def test_vta_plugin_registers_readout_and_learning_through_session():
    session, _, vta, _, _ = _build_vta(1)

    assert vta.dopamine in session.observers
    assert vta.stdp in session.observers
    assert vta.stdp in session.adaptations


def _build_vta(seed: int):
    builder = NetworkBuilder()
    positive = builder.add_population("OUTCOME_POSITIVE", 16)
    negative = builder.add_population("OUTCOME_NEGATIVE", 16, output="inhibitory")
    vta_handle = add_vta_dopamine(builder)
    for outcome in (positive, negative):
        builder.connect(
            outcome,
            vta_handle.dopamine_input,
            FanInSpec(8),
            StrengthSpec(0.30, 0.02, maximum=0.8),
            "fixed",
            False,
        )
    session = builder.compile(seed)
    populations = {
        population.spec.name: population for population in session.snn.layout.populations
    }
    session.add(
        UnipolarRateInput(populations[positive.name], np.random.default_rng([seed, 1])),
        UnipolarRateInput(populations[negative.name], np.random.default_rng([seed, 2])),
    )
    positive_input, negative_input = session.drive_sources[-2:]
    assert isinstance(positive_input, UnipolarRateInput)
    assert isinstance(negative_input, UnipolarRateInput)
    return (
        session,
        populations,
        compiled_vta_dopamine(session, vta_handle),
        positive_input,
        negative_input,
    )


def _tick(session, ticks: int) -> None:
    for _ in range(ticks):
        session.tick()


def _modulation(session, vta, ticks: int) -> list[float]:
    values = []
    for _ in range(ticks):
        session.tick()
        values.append(vta.dopamine.value)
    return values
