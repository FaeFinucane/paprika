"""Contract for rate inputs and population readouts."""

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.interaction.drives import TonicDrive
from src.interaction.rates import (
    DopamineReadout,
    PopulationRate,
    RatePatternInput,
    UnipolarRateInput,
)
from src.network.population import NeuronPopulationSpec, PopulationLayout
from src.network.snn import Spikes

pytestmark = pytest.mark.unit


def population():
    return PopulationLayout.build((NeuronPopulationSpec("x", 4),)).populations[0]


def spikes(pop, active):
    values = np.zeros(pop.spec.count, dtype=bool)
    values[:active] = True
    return Spikes(values, 1, pop.layout_fingerprint)


def test_rate_input_is_bounded_and_emits_for_exact_duration():
    pop = population()
    source = UnipolarRateInput(pop, np.random.default_rng(2), duration=2)
    source.write(0.75)
    assert source.produce().drives[pop].shape == (4,)
    assert source.produce().drives[pop].shape == (4,)
    assert source.produce().drives == {}
    with np.testing.assert_raises(ValueError):
        source.write(1.1)


def test_rate_input_clear_and_reset_cancel_pending_drive():
    pop = population()
    source = UnipolarRateInput(pop, duration=3)
    source.write(1.0)
    source.clear()
    assert source.produce().drives == {}
    source.write(1.0)
    source.reset()
    assert source.produce().drives == {}


def test_pattern_input_drives_a_bounded_rate_vector():
    pop = population()
    source = RatePatternInput(pop, np.random.default_rng(2))
    source.write(np.array([0.0, 1.0, 1.0, 0.0]))
    assert np.array_equal(source.produce().drives[pop], np.array([0.0, 1.0, 1.0, 0.0]))


def test_tonic_drive_has_fixed_zero_mean_cell_diversity():
    pop = population()
    source = TonicDrive(pop, 0.4, heterogeneity=0.1, rng=np.random.default_rng(3))
    first = source.produce().drives[pop]
    second = source.produce().drives[pop]
    assert np.isclose(float(np.mean(first)), 0.4)
    assert not np.allclose(first, first[0])
    assert np.array_equal(first, second)


def test_population_rate_decodes_monotonically_and_resets():
    pop = population()
    decoder = PopulationRate(pop, decay=0.5)
    decoder.observe(spikes(pop, 1))
    low = decoder.rate
    decoder.observe(spikes(pop, 4))
    high = decoder.rate
    assert 0.0 < low < high <= 1.0
    decoder.reset()
    assert decoder.rate == 0.0
    assert decoder.report_name == "rate:x"


def test_dopamine_baseline_and_symmetric_deadzone():
    pop = population()
    rate = PopulationRate(pop, decay=0.0)
    dopamine = DopamineReadout(rate, baseline=0.5, deadzone=0.1)
    assert dopamine.decode(0.5) == 0.0
    assert dopamine.decode(0.55) == 0.0
    assert np.isclose(dopamine.decode(0.8), -dopamine.decode(0.2))
    zero_baseline = DopamineReadout(rate, baseline=0.0, deadzone=0.1)
    assert zero_baseline.decode(0.0) == 0.0
    assert dopamine.report_name == "dopamine"


def test_dopamine_readout_decodes_the_rate_observed_once_by_the_session():
    builder = NetworkBuilder()
    builder.add_population("DA", 4)
    session = builder.compile(1)
    population = session.snn.layout.population("DA")
    rate = PopulationRate(population, decay=0.5)
    dopamine = DopamineReadout(rate, baseline=0.25, deadzone=0.0)
    session.add(rate, dopamine)
    session.snn.neurons.voltage[:] = 3.0

    session.tick()

    assert rate.rate == 0.5
    assert dopamine.value > 0.0
