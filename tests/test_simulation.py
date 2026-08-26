import numpy as np
import pytest

from continual_agent.simulation import LIFNeurons, SparseSynapses, SpikingNetwork


def test_lif_neuron_spikes_and_resets() -> None:
    neurons = LIFNeurons(count=1, dt=1.0, tau_membrane=10.0, threshold=1.0)

    spikes = [neurons.step(np.array([2.0]))[0] for _ in range(10)]

    assert any(spikes)
    assert neurons.voltage[0] < neurons.threshold


def test_lif_refractory_period_suppresses_spikes() -> None:
    neurons = LIFNeurons(
        count=1, dt=1.0, tau_membrane=1.0, threshold=1.0, refractory_ticks=2
    )

    spikes = [bool(neurons.step(np.array([2.0]))[0]) for _ in range(5)]

    assert spikes == [True, False, False, True, False]


def test_sparse_synapses_transmit_only_from_active_sources() -> None:
    synapses = SparseSynapses(
        source=np.array([0, 1, 1]),
        target=np.array([2, 2, 3]),
        weight=np.array([0.5, 0.25, -0.5]),
        neuron_count=4,
    )

    current = synapses.transmit(np.array([False, True, False, False]))

    np.testing.assert_allclose(current, [0.0, 0.0, 0.25, -0.5])


def test_random_network_is_reproducible() -> None:
    first = SpikingNetwork.random(neuron_count=20, seed=42, connection_probability=0.2)
    second = SpikingNetwork.random(neuron_count=20, seed=42, connection_probability=0.2)
    external = np.zeros(20)
    external[0] = 2.0

    first_spikes = [first.step(external) for _ in range(20)]
    second_spikes = [second.step(external) for _ in range(20)]

    np.testing.assert_array_equal(first.synapses.source, second.synapses.source)
    np.testing.assert_array_equal(first.synapses.target, second.synapses.target)
    np.testing.assert_array_equal(first_spikes, second_spikes)


def test_recurrent_network_remains_finite() -> None:
    network = SpikingNetwork.random(neuron_count=50, seed=7, connection_probability=0.05)

    for _ in range(200):
        network.step(np.zeros(50))

    assert np.isfinite(network.neurons.voltage).all()
    assert np.isfinite(network.synapses.weight).all()
    assert len(network.spike_history) == 200


def test_shape_mismatches_are_rejected() -> None:
    neurons = LIFNeurons(count=2)
    with pytest.raises(ValueError):
        neurons.step(np.zeros(1))
