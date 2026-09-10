import numpy as np
import pytest
from src.network.definition import NetworkDefinition
from src.network.population import NeuronPopulationSpec

from tests.support.perturbations import (
    add_membrane_voltage,
    clear_transient_activity,
    set_membrane_voltage,
    silence_neurons,
)

pytestmark = pytest.mark.unit


def test_perturbations_change_only_transient_neuron_state():
    snn = NetworkDefinition((NeuronPopulationSpec("POPULATION", 4),), ()).compile(
        np.random.default_rng(1)
    )
    population = snn.layout.population("POPULATION")
    snn.next_current[:] = 0.3
    initial_strengths = snn.synapses.strength.copy()

    set_membrane_voltage(snn, population, 0.4, count=2)
    add_membrane_voltage(snn, population, 0.1, count=2)
    assert np.allclose(snn.neurons.voltage, [0.5, 0.5, 0.0, 0.0])

    silence_neurons(snn, population, count=2)
    assert np.allclose(snn.neurons.voltage, [-0.5, -0.5, 0.0, 0.0])
    assert np.allclose(snn.next_current[:2], 0.0)
    assert np.allclose(snn.next_current[2:], 0.3)

    clear_transient_activity(snn)
    assert np.allclose(snn.next_current, 0.0)
    assert np.array_equal(snn.synapses.strength, initial_strengths)
