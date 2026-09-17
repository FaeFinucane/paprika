"""Contract for fixed cue-to-outcome experiment reports."""

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.circuits.vta_dopamine import add_vta_dopamine_circuit
from src.experiments.dopamine import FixedCueOutcomeProtocol, run_fixed_cue_outcome
from src.experiments.recording import ProjectionStrengthSnapshot

pytestmark = pytest.mark.unit


def test_fixed_protocol_automatically_records_dopamine_and_requested_rates():
    builder = NetworkBuilder()
    declaration = add_vta_dopamine_circuit(builder)
    readout = builder.connection(declaration.temporal.excitatory, declaration.vta_inhibitory)
    protocol = FixedCueOutcomeProtocol(
        trials=2,
        settle_ticks=0,
        cue_ticks=2,
        cue_to_outcome_ticks=1,
        outcome_ticks=3,
        inter_trial_ticks=0,
    )

    report = run_fixed_cue_outcome(
        declaration,
        protocol,
        seeds=(2, 3),
        population_rates=(declaration.vta_inhibitory,),
        snapshots=(ProjectionStrengthSnapshot(readout),),
        enable_dopamine_learning=False,
    )

    assert report.dopamine.shape == (2, 2, protocol.event_ticks)
    assert report.population_rate(declaration.vta_inhibitory).shape == (2, 2, protocol.event_ticks)
    assert report.stream("rate:VTA_DA").shape == (2, 2, protocol.event_ticks)
    strengths = report.snapshot("strength:TEMPORAL_E_to_VTA_INHIB")
    assert strengths.shape == (2, protocol.trials + 2, 3)
    assert np.all(strengths[..., 0] <= strengths[..., 1])
    assert np.all(strengths[..., 1] <= strengths[..., 2])


def test_fixed_protocol_rejects_population_rate_already_registered_by_the_circuit():
    declaration = add_vta_dopamine_circuit(NetworkBuilder())

    with pytest.raises(ValueError, match="already observes"):
        run_fixed_cue_outcome(
            declaration,
            FixedCueOutcomeProtocol(trials=1, settle_ticks=0, inter_trial_ticks=0),
            population_rates=(declaration.vta_dopamine,),
            enable_dopamine_learning=False,
        )
