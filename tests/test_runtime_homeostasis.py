import numpy as np

from continual_agent.agent.population_homeostasis import HomeostasisConfig, PopulationHomeostasis
from continual_agent.agent.runtime_metrics import RuntimeMetrics
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.simulation.population_layout import Population, PopulationLayout
from tests.helpers import network_config


def test_population_metrics_define_activity_fractions_and_distributions() -> None:
    layout = PopulationLayout(
        input_count=0, hidden_count=2, affect_count=0, action_count=0, char_count=0
    )
    metrics = RuntimeMetrics(layout, saturation_rate=0.5)
    for voltage, spikes in (([0.2, 0.4], [1, 0]), ([0.3, 0.5], [1, 0])):
        metrics.record(np.asarray(spikes), np.asarray(voltage))

    result = metrics.population(Population.HIDDEN, 1.0).as_dict()
    assert result["neuron_count"] == 2.0
    assert result["spike_count"] == 2.0
    assert result["firing_rate_mean"] == 0.5
    assert result["firing_rate_spread"] == 0.5
    assert result["active_fraction"] == 0.5
    assert result["silent_fraction"] == 0.5
    assert result["saturated_fraction"] == 0.5
    assert result["voltage_min"] == 0.2
    assert result["voltage_max"] == 0.5
    assert result["integrated_voltage"] == 1.4
    assert result["threshold_mean"] == 1.0
    metrics.record_output_event()
    assert metrics.output_event_rate == 0.5


def test_homeostasis_moves_toward_target_and_is_bounded() -> None:
    layout = PopulationLayout(
        input_count=0, hidden_count=2, affect_count=0, action_count=0, char_count=0
    )
    controller = PopulationHomeostasis(
        layout,
        HomeostasisConfig(
            enabled=True, target_rate=0.25, strength=1.0, update_interval=1, max_current=0.2
        ),
    )
    controller.observe(np.array([0.0, 0.0]))
    assert controller.drive[0] == 0.2
    controller.observe(np.array([1.0, 1.0]))
    assert controller.drive[0] < 0.2
    assert controller.drive[0] >= -0.2
    controller.observe(np.array([1.0, 1.0]))
    assert controller.drive[0] == -0.2
    assert np.all(np.abs(controller.drive) <= 0.2)


def test_homeostasis_does_not_change_stdp_specialization() -> None:
    runtime = SpikingRuntime(
        network_config(
            input_features=4,
            hidden_neurons=4,
            output_tokens=("<EOS>", "A"),
            neurons_per_token=1,
            seed=7,
            homeostasis=HomeostasisConfig(enabled=True, update_interval=1),
        )
    )
    before = runtime.network.synapses.weight.copy()
    for _ in range(8):
        runtime.step(np.zeros(runtime.network.neurons.count))
    np.testing.assert_array_equal(runtime.network.synapses.weight, before)
    assert np.any(runtime.homeostasis.drive != 0.0)


def test_enabled_homeostasis_is_deterministic_for_a_seed() -> None:
    def make_runtime() -> SpikingRuntime:
        return SpikingRuntime(
            network_config(
                input_features=4,
                hidden_neurons=4,
                output_tokens=("<EOS>", "A"),
                neurons_per_token=1,
                seed=11,
                background_rate=0.5,
                background_current=0.4,
                homeostasis=HomeostasisConfig(enabled=True, update_interval=2),
            )
        )

    first = make_runtime()
    second = make_runtime()
    current = np.zeros(first.network.neurons.count)
    for _ in range(12):
        np.testing.assert_array_equal(first.step(current), second.step(current))
    np.testing.assert_array_equal(first.homeostasis.drive, second.homeostasis.drive)
