"""Behavioral contracts for the standalone recurrent ramp circuit."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.interaction.plasticity import DopamineSTDP
from src.interaction.rates import PopulationRate, PopulationRateSpec, RatePatternInput
from src.network.connectivity import FanInSpec, StrengthSpec

from .ramping import RampingCircuitHandle, RampingCircuitSpec, add_ramping_circuit

pytestmark = pytest.mark.viability


@dataclass
class TeachingDopamine:
    """Explicit test-only DA signal used to assess existing plasticity."""

    value: float = 0.0


def test_ramping_circuit_has_a_bounded_rise_peak_and_decay_after_cue_withdrawal():
    traces = [_response_trace(seed) for seed in range(4)]
    ramp = np.mean(traces, axis=0)
    peak = int(np.argmax(ramp))

    assert 5 <= peak <= 9
    assert ramp[peak] >= 0.55
    assert np.all(np.diff(ramp[1 : peak + 1]) >= -0.03)
    assert ramp[16] <= 0.25
    assert ramp[-1] <= 0.05


def test_negative_dopamine_at_a_premature_peak_delays_the_ramp_peak():
    before, after = _train_peak(lambda tick: -0.7 if tick == 7 else 0.0)

    assert int(np.argmax(after)) >= int(np.argmax(before)) + 1
    assert after[10] > before[10]


def test_positive_dopamine_after_the_ramp_suppresses_late_ramp_activity():
    before, after = _train_peak(lambda tick: 0.7 if tick == 10 else 0.0)

    assert after[10] < before[10]


def test_ramping_circuit_exposes_a_single_ordinary_learnable_termination_path():
    builder, _, handle = _declare_ramp()

    assert isinstance(handle, RampingCircuitHandle)
    assert builder.connection(handle.excitatory, handle.inhibitory).learning == "dopamine_stdp"
    assert builder.connection(handle.excitatory, handle.excitatory).learning == "fixed"


def _response_trace(seed: int) -> np.ndarray:
    session, populations, _, cue_input, rate, dopamine = _compile_ramp(seed)
    return _present_cue(session, populations, cue_input, rate, dopamine)


def _train_peak(signal) -> tuple[np.ndarray, np.ndarray]:
    before_traces = []
    after_traces = []
    for seed in range(4):
        session, populations, _, cue_input, rate, dopamine = _compile_ramp(seed)
        before_traces.append(_present_cue(session, populations, cue_input, rate, dopamine))
        for _ in range(100):
            _present_cue(session, populations, cue_input, rate, dopamine, signal)
        _tick(session, 30)
        after_traces.append(_present_cue(session, populations, cue_input, rate, dopamine))
    return np.mean(before_traces, axis=0), np.mean(after_traces, axis=0)


def _declare_ramp():
    builder = NetworkBuilder()
    cue = builder.add_feature_population("CUE", ("CUE",), width=16)
    handle = add_ramping_circuit(builder, RampingCircuitSpec())
    builder.connect(
        cue,
        handle.input,
        FanInSpec(8),
        StrengthSpec(0.50, 0.02, maximum=0.8),
        "fixed",
        False,
    )
    return builder, cue, handle


def _compile_ramp(seed: int):
    builder, cue, handle = _declare_ramp()
    session = builder.compile(seed)
    populations = {
        population.spec.name: population for population in session.snn.layout.populations
    }
    cue_input = RatePatternInput(populations[cue.name], np.random.default_rng([seed, 1]))
    rate = PopulationRate(populations[handle.excitatory], PopulationRateSpec(decay=0.8))
    dopamine = TeachingDopamine()
    session.add(cue_input, rate, DopamineSTDP(session.snn, dopamine, learning_rate=0.0005))
    return session, populations, handle, cue_input, rate, dopamine


def _present_cue(session, populations, cue_input, rate, dopamine, signal=None) -> np.ndarray:
    pattern = np.zeros(populations["CUE"].count)
    pattern[: pattern.size // 2] = 1.0
    cue_input.write(pattern, 5)
    trace = []
    for tick in range(30):
        dopamine.value = 0.0 if signal is None else float(signal(tick))
        session.tick()
        trace.append(rate.value)
    dopamine.value = 0.0
    return np.asarray(trace)


def _tick(session, ticks: int) -> None:
    for _ in range(ticks):
        session.tick()
