"""Multi-seed behavioural contract for a minimal recurrent E/I circuit."""

import numpy as np
import pytest
from src.diagnostics import NetworkTrace, inspect_network
from src.interaction.drives import HomeostaticDrive
from src.interaction.plasticity.homeostasis import SynapticScaling
from src.interaction.plasticity.inhibitory import InhibitoryHomeostasis
from src.network.connectivity import ConnectionSpec, FanInSpec, StrengthSpec
from src.network.definition import NetworkDefinition
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
        session.adaptations.clear()
        session.stateful_adaptations.clear()

        # One whole-network E perturbation should recruit I and contain E over
        # the ensuing short causal response window.
        clear_transient_activity(snn)
        set_membrane_voltage(snn, excitatory, voltage=3.0)
        stimulated = session.tick()
        response = [session.tick() for _ in range(3)]
        assert float(np.mean(stimulated.population(excitatory))) == 1.0
        assert max(float(np.mean(spikes.population(inhibitory))) for spikes in response) >= baseline_i + 0.25
        assert np.mean([np.mean(spikes.population(excitatory)) for spikes in response]) <= 0.20

        for _ in range(300):
            session.tick()
        assert 0.04 <= _tail_rate(trace, "EXCITATORY", 100) <= 0.16

        # Withdrawal intentionally permits either quiescence or sparse
        # persistence. Only sustained high activity is a generic failure.
        for drive in homeostatic_drives:
            drive.enabled = False
        withdrawal_e: list[float] = []
        for _ in range(500):
            spikes = session.tick()
            withdrawal_e.append(float(np.mean(spikes.population(excitatory))))
        assert max(_window_means(withdrawal_e, 50)) <= 0.50
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
    definition = NetworkDefinition(
        populations=(
            NeuronPopulationSpec("EXCITATORY", 64),
            NeuronPopulationSpec("INHIBITORY", 16, output="inhibitory"),
        ),
        connections=(
            ConnectionSpec(
                "EXCITATORY",
                "EXCITATORY",
                FanInSpec(8),
                StrengthSpec(0.10, maximum=0.40),
                learning="homeostatic",
                scalable=True,
            ),
            ConnectionSpec(
                "EXCITATORY",
                "INHIBITORY",
                FanInSpec(12),
                StrengthSpec(0.15, maximum=0.50),
            ),
            ConnectionSpec(
                "INHIBITORY",
                "EXCITATORY",
                FanInSpec(16),
                StrengthSpec(0.20, maximum=0.80),
                learning="inhibitory_homeostatic",
            ),
        ),
    )
    snn = definition.compile(np.random.default_rng(seed))
    excitatory = snn.layout.population("EXCITATORY")
    inhibitory = snn.layout.population("INHIBITORY")
    excitatory_drive = HomeostaticDrive(
        excitatory,
        target_rate=0.08,
        learning_rate=0.002,
        rate_decay=0.98,
        minimum_current=-0.5,
        maximum_current=0.5,
    )
    inhibitory_drive = HomeostaticDrive(
        inhibitory,
        target_rate=0.10,
        learning_rate=0.002,
        rate_decay=0.98,
        minimum_current=-0.5,
        maximum_current=0.5,
    )
    scaling = SynapticScaling(snn, excitatory, target_rate=0.08, learning_rate=0.0001, rate_decay=0.995)
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


def _tail_rate(trace: NetworkTrace, population: str, ticks: int) -> float:
    return float(np.mean([sample.population_rates[population] for sample in trace.samples[-ticks:]]))


def _window_means(values: list[float], width: int) -> list[float]:
    if width <= 0 or len(values) < width:
        raise ValueError("window width must fit within supplied values")
    return [float(np.mean(values[start : start + width])) for start in range(len(values) - width + 1)]
