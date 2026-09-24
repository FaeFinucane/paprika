"""Contract for fixed cue-to-outcome experiment reports."""

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.experiments.dopamine import (
    FixedCueOutcomeProtocol,
    add_fixed_cue_outcome_conditioning,
    run_fixed_cue_outcome,
)
from src.experiments.recording import ProjectionStrengthSnapshot

pytestmark = pytest.mark.unit


@pytest.mark.viability
def test_fixed_conditioning_acquisition_has_an_outcome_aligned_vta_inhibitory_peak():
    early_inhibitory = []
    late_inhibitory = []
    early_dopamine = []
    late_dopamine = []
    for seed in range(4):
        circuit = add_fixed_cue_outcome_conditioning(NetworkBuilder()).build(seed)
        _tick(circuit, 1_000)
        inhibitory, dopamine = _paired_traces(circuit, trials=100)
        early_inhibitory.extend(inhibitory[:20])
        late_inhibitory.extend(inhibitory[-20:])
        early_dopamine.extend(dopamine[:20])
        late_dopamine.extend(dopamine[-20:])

    early_peak = int(np.argmax(np.mean(early_inhibitory, axis=0)))
    late_peak = int(np.argmax(np.mean(late_inhibitory, axis=0)))
    assert 9 <= early_peak <= 12
    assert 9 <= late_peak <= 12

    early_reward = float(np.max(np.mean(early_dopamine, axis=0)[10:]))
    late_reward = float(np.max(np.mean(late_dopamine, axis=0)[10:]))
    late_cue = float(np.max(np.mean(late_dopamine, axis=0)[:10]))
    assert late_reward <= 0.5 * early_reward
    assert late_cue >= 0.2 * early_reward


def test_fixed_protocol_automatically_records_dopamine_and_requested_rates():
    builder = NetworkBuilder()
    declaration = add_fixed_cue_outcome_conditioning(builder)
    source = declaration.temporal.stages[-1]
    readout = builder.connection(source, declaration.vta.inhibitory_input)
    protocol = FixedCueOutcomeProtocol(
        trials=2,
        settle_ticks=0,
        cue_ticks=2,
        cue_to_outcome_ticks=1,
        outcome_ticks=3,
        response_tail_ticks=2,
        inter_trial_ticks=0,
    )

    report = run_fixed_cue_outcome(
        declaration,
        protocol,
        seeds=(2, 3),
        population_rates=(declaration.vta.inhibitory, source),
        snapshots=(ProjectionStrengthSnapshot(readout),),
        enable_dopamine_learning=False,
    )

    assert report.dopamine.shape == (2, 2, protocol.event_ticks)
    assert report.population_rate(declaration.vta.inhibitory).shape == (2, 2, protocol.event_ticks)
    assert report.population_rate(source).shape == (
        2,
        2,
        protocol.event_ticks,
    )
    assert report.stream("rate:VTA_DA").shape == (2, 2, protocol.event_ticks)
    strengths = report.snapshot(f"strength:{source}_to_VTA_INHIB")
    assert strengths.shape == (2, protocol.trials + 2, 3)
    assert np.all(strengths[..., 0] <= strengths[..., 1])
    assert np.all(strengths[..., 1] <= strengths[..., 2])


def test_fixed_protocol_rejects_population_rate_already_registered_by_the_circuit():
    declaration = add_fixed_cue_outcome_conditioning(NetworkBuilder())

    with pytest.raises(ValueError, match="already observes"):
        run_fixed_cue_outcome(
            declaration,
            FixedCueOutcomeProtocol(trials=1, settle_ticks=0, inter_trial_ticks=0),
            population_rates=(declaration.vta.dopamine,),
            enable_dopamine_learning=False,
        )


def test_fixed_circuit_removes_dopamine_rule_for_a_nonlearning_run():
    circuit = add_fixed_cue_outcome_conditioning(NetworkBuilder()).build(
        1, enable_dopamine_learning=False
    )

    assert all(adaptation is not circuit.stdp for adaptation in circuit.session.adaptations)
    assert all(observer is not circuit.stdp for observer in circuit.session.observers)


def _tick(circuit, ticks: int) -> None:
    for _ in range(ticks):
        circuit.tick()


def _paired_traces(circuit, *, trials: int) -> tuple[list[list[float]], list[list[float]]]:
    cue = np.zeros(circuit.populations["CUE"].count)
    cue[: cue.size // 2] = 1.0
    inhibitory_traces = []
    dopamine_traces = []
    for _ in range(trials):
        circuit.present_cue(cue, 5)
        inhibitory = []
        dopamine = []
        for _ in range(10):
            spikes = circuit.tick()
            inhibitory.append(float(np.mean(spikes.population(circuit.vta.inhibitory))))
            dopamine.append(circuit.dopamine.value)
        circuit.deliver_outcome(0.7, 10)
        for _ in range(10):
            spikes = circuit.tick()
            inhibitory.append(float(np.mean(spikes.population(circuit.vta.inhibitory))))
            dopamine.append(circuit.dopamine.value)
        inhibitory_traces.append(inhibitory)
        dopamine_traces.append(dopamine)
        _tick(circuit, 30)
    return inhibitory_traces, dopamine_traces
