"""Multi-seed behavioural contract for a minimal recurrent E/I circuit."""

import numpy as np
import pytest
from src.builder import NetworkBuilder
from src.diagnostics import NetworkTrace, inspect_network
from src.interaction.drives import HomeostaticDrive, HomeostaticDriveSpec
from src.interaction.plasticity.homeostasis import SynapticScaling, SynapticScalingSpec
from src.interaction.plasticity.inhibitory import InhibitoryHomeostasis
from src.interaction.rates import UnipolarRateInput
from src.network.connectivity import FanInSpec, FanOutSpec, StrengthSpec
from src.network.population import NeuronPopulationSpec, Population
from src.network.snn import SNN
from src.session import Session

from tests.support.perturbations import (
    clear_transient_activity,
    set_membrane_voltage,
)

pytestmark = pytest.mark.viability


def test_minimal_recurrent_ei_circuit_has_a_bounded_feedback_response_across_seeds():
    """The assembled E/I loop is responsive, contained, and safe on withdrawal."""
    for seed in range(8):
        snn, session, trace, excitatory, inhibitory, homeostatic_drives = _minimal_ei_system(seed)

        for _ in range(3000):
            session.tick()
        baseline_e = _tail_rate(trace, "EXCITATORY", 300)
        baseline_i = _tail_rate(trace, "INHIBITORY", 300)
        assert 0.04 <= baseline_e <= 0.16
        assert 0.02 <= baseline_i <= 0.25
        assert inspect_network(snn).valid

        # Freeze slow adaptation at its selected operating point. The explicit
        # drives remain active, but cannot repair the fast response below.
        with session.frozen_adaptations():
            # One whole-network E perturbation should recruit I and contain E
            # over the ensuing short causal response window.
            clear_transient_activity(snn)
            set_membrane_voltage(snn, excitatory, voltage=3.0)
            stimulated = session.tick()
            response = [session.tick() for _ in range(3)]
            assert float(np.mean(stimulated.population(excitatory))) == 1.0
            assert (
                max(float(np.mean(spikes.population(inhibitory))) for spikes in response)
                >= baseline_i + 0.25
            )
            assert np.mean([np.mean(spikes.population(excitatory)) for spikes in response]) <= 0.20

        for _ in range(300):
            session.tick()
        assert 0.04 <= _tail_rate(trace, "EXCITATORY", 100) <= 0.16

        # Withdrawal intentionally permits either quiescence or sparse
        # persistence. Only sustained high activity is a generic failure.
        session.remove(*homeostatic_drives)
        withdrawal_e: list[float] = []
        for _ in range(500):
            spikes = session.tick()
            withdrawal_e.append(float(np.mean(spikes.population(excitatory))))
        assert max(_window_means(withdrawal_e, 50)) <= 0.50
        assert inspect_network(snn).valid


def test_minimal_ei_circuit_transmits_distinct_input_patterns_across_seeds():
    """A recurrent E/I substrate carries distinct inputs to a downstream population.

    This is deliberately a whole-circuit behavioural contract: the two source
    patterns have no direct projection to OUTPUT, and the test makes no claim
    about which individual output neurons should represent either pattern.
    """
    for seed in range(8):
        snn, session, input_a, input_b, output = _ei_transmission_system(seed)
        for _ in range(3_000):
            session.tick()

        # Fix slow regulators at their selected operating point. Repeating a
        # fully active rate-coded pattern from the same transient state gives
        # this first contract a clean, encoder-noise-free fidelity floor.
        with session.frozen_adaptations():
            baseline = _response_trials(snn, session, None, output)
            response_a = _response_trials(snn, session, input_a, output)
            response_b = _response_trials(snn, session, input_b, output)

        mean_baseline = np.mean(baseline, axis=0)
        mean_a = np.mean(response_a, axis=0)
        mean_b = np.mean(response_b, axis=0)
        input_effect = min(
            np.linalg.norm(mean_a - mean_baseline),
            np.linalg.norm(mean_b - mean_baseline),
        )
        separation = np.linalg.norm(mean_a - mean_b)
        within_pattern_variation = max(
            _mean_distance_to_centroid(response_a, mean_a),
            _mean_distance_to_centroid(response_b, mean_b),
        )

        # The downstream response must both move away from its unprompted
        # state and preserve the distinction between equally sized inputs.
        # These margins reject a silent, saturated, or pattern-collapsing
        # output without prescribing which output neurons encode a pattern.
        assert input_effect >= 0.12
        assert separation >= 0.12
        assert within_pattern_variation <= 0.01
        assert max(float(np.mean(response_a)), float(np.mean(response_b))) <= 0.20
        assert inspect_network(snn).valid


def _minimal_ei_system(
    seed: int,
) -> tuple[
    SNN,
    Session,
    NetworkTrace,
    Population[NeuronPopulationSpec],
    Population[NeuronPopulationSpec],
    tuple[HomeostaticDrive, HomeostaticDrive],
]:
    builder = NetworkBuilder()
    builder.add_population("EXCITATORY", 64)
    builder.add_population("INHIBITORY", 16, output="inhibitory")
    builder.connect(
        "EXCITATORY",
        "EXCITATORY",
        FanInSpec(8),
        StrengthSpec(0.10, maximum=0.40),
        learning="homeostatic",
        scalable=True,
    )
    builder.connect("EXCITATORY", "INHIBITORY", FanInSpec(12), StrengthSpec(0.15, maximum=0.50))
    builder.connect(
        "INHIBITORY",
        "EXCITATORY",
        FanInSpec(16),
        StrengthSpec(0.20, maximum=0.80),
        learning="inhibitory_homeostatic",
    )
    snn = builder.compile(seed).snn
    excitatory = snn.layout.population("EXCITATORY")
    inhibitory = snn.layout.population("INHIBITORY")
    excitatory_drive = HomeostaticDrive(
        excitatory,
        HomeostaticDriveSpec(
            target_rate=0.08,
            learning_rate=0.002,
            rate_decay=0.98,
            minimum_current=-0.5,
            maximum_current=0.5,
        ),
    )
    inhibitory_drive = HomeostaticDrive(
        inhibitory,
        HomeostaticDriveSpec(
            target_rate=0.10,
            learning_rate=0.002,
            rate_decay=0.98,
            minimum_current=-0.5,
            maximum_current=0.5,
        ),
    )
    scaling = SynapticScaling(
        snn,
        excitatory,
        SynapticScalingSpec(target_rate=0.08, learning_rate=0.0001, rate_decay=0.995),
    )
    inhibitory_homeostasis = InhibitoryHomeostasis(
        snn, target_rate=0.08, learning_rate=0.0005, trace_decay=0.95
    )
    trace = NetworkTrace(
        snn,
        [excitatory, inhibitory],
        current_drives=[excitatory_drive, inhibitory_drive],
        include_projection_strengths=False,
    )
    session = Session.build(
        snn,
        [excitatory_drive, inhibitory_drive, scaling, inhibitory_homeostasis, trace],
    )
    return snn, session, trace, excitatory, inhibitory, (excitatory_drive, inhibitory_drive)


def _ei_transmission_system(
    seed: int,
) -> tuple[SNN, Session, UnipolarRateInput, UnipolarRateInput, Population[NeuronPopulationSpec]]:
    """Build a minimal feed-forward-through-recurrent-E/I signal path."""
    builder = NetworkBuilder()
    builder.add_feature_population("INPUT_A", ("A",), width=8)
    builder.add_feature_population("INPUT_B", ("B",), width=8)
    builder.add_population("EXCITATORY", 64)
    builder.add_population("INHIBITORY", 16, output="inhibitory")
    builder.add_population("OUTPUT", 32)
    # Equivalent sparse input channels seed independently sampled E assemblies.
    # There is intentionally no INPUT -> OUTPUT shortcut.
    builder.connect("INPUT_A", "EXCITATORY", FanOutSpec(6, target_neurons=16), StrengthSpec(0.30))
    builder.connect("INPUT_B", "EXCITATORY", FanOutSpec(6, target_neurons=16), StrengthSpec(0.30))
    builder.connect("EXCITATORY", "EXCITATORY", FanInSpec(8), StrengthSpec(0.10))
    builder.connect("EXCITATORY", "INHIBITORY", FanInSpec(12), StrengthSpec(0.15))
    builder.connect("INHIBITORY", "EXCITATORY", FanInSpec(16), StrengthSpec(0.20))
    builder.connect("EXCITATORY", "OUTPUT", FanInSpec(16), StrengthSpec(0.18))
    rng = np.random.default_rng(seed)
    snn = builder.compile(seed).snn
    input_a = UnipolarRateInput(snn.layout.population("INPUT_A"), rng)
    input_b = UnipolarRateInput(snn.layout.population("INPUT_B"), rng)
    excitatory = snn.layout.population("EXCITATORY")
    inhibitory = snn.layout.population("INHIBITORY")
    output = snn.layout.population("OUTPUT")
    components = [
        input_a,
        input_b,
        HomeostaticDrive(
            excitatory,
            HomeostaticDriveSpec(
                target_rate=0.08,
                learning_rate=0.002,
                rate_decay=0.98,
                minimum_current=-0.5,
                maximum_current=0.5,
            ),
        ),
        HomeostaticDrive(
            inhibitory,
            HomeostaticDriveSpec(
                target_rate=0.10,
                learning_rate=0.002,
                rate_decay=0.98,
                minimum_current=-0.5,
                maximum_current=0.5,
            ),
        ),
        HomeostaticDrive(
            output,
            HomeostaticDriveSpec(
                target_rate=0.05,
                learning_rate=0.002,
                rate_decay=0.98,
                minimum_current=-0.5,
                maximum_current=0.5,
            ),
        ),
    ]
    return snn, Session.build(snn, components), input_a, input_b, output


def _response_trials(
    snn: SNN,
    session: Session,
    source: UnipolarRateInput | None,
    output: Population[NeuronPopulationSpec],
    *,
    trials: int = 12,
    stimulus_ticks: int = 8,
    response_ticks: int = 12,
) -> np.ndarray:
    responses: list[np.ndarray] = []
    for _ in range(trials):
        clear_transient_activity(snn)
        if source is not None:
            source.write(1.0, stimulus_ticks)
        samples: list[np.ndarray] = []
        for _ in range(response_ticks):
            spikes = session.tick()
            samples.append(spikes.population(output).astype(float))
        responses.append(np.mean(samples, axis=0))
    return np.asarray(responses)


def _mean_distance_to_centroid(samples: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(samples - centroid, axis=1)))


def _tail_rate(trace: NetworkTrace, population: str, ticks: int) -> float:
    return float(
        np.mean([sample.population_rates[population] for sample in trace.samples[-ticks:]])
    )


def _window_means(values: list[float], width: int) -> list[float]:
    if width <= 0 or len(values) < width:
        raise ValueError("window width must fit within supplied values")
    return [
        float(np.mean(values[start : start + width])) for start in range(len(values) - width + 1)
    ]
