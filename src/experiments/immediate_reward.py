"""Minimal test: Using STDP and a fixed external reward
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..interaction.background import BackgroundDrive
from ..interaction.channels import FeatureInChannel, FeatureOutChannel, PopulationDecoder, PopulationEncoder
from ..interaction.plasticity import Hebbian, InhibitoryPlasticity
from ..network.connectivity import FanOutSpec, WeightSpec, ConnectionSpec, Connectivity
from ..network.population import FeaturePopulationSpec, NeuronPopulationSpec, PopulationLayout
from ..network.snn import SNN
from ..session import Session
from .curriculum import duration_reward, ramp

PROMPT_FEATURE = "CUE"
REWARD_VALUE = 0.75
IDEAL_DURATION = 3
# Lower means (down to ~0.25) also learn fine and show clearer resp% curves,
# but 0.6 is kept as the baseline for future, more complex training.
DEFAULT_WEIGHT = WeightSpec(0.6, 0.04)
# Seeded much weaker than DEFAULT_WEIGHT so the network can still fall quiet
# between responses instead of self-sustaining - see reward_shaping.py.
RECURRENT_SEED_WEIGHT = WeightSpec(0.08, 0.01)

@dataclass
class ExternalRPE:
    """A stand-in for the internal plasticity.RPE to determine reward value externally."""
    value: float = field(init=False, default=0.0)

    def notify(self, value: float):
        self.value = value

@dataclass
class Trial:
    start_tick: int
    end_tick: int
    responded: bool
    response_tick: int | None = None
    duration: int | None = None
    reward_value: float | None = None
    rpe: float | None = None
    violations: int = 0


@dataclass
class EpochStats:
    index: int
    start_tick: int
    end_tick: int
    trials: int
    responded: int
    mean_response_latency: float | None
    mean_duration: float | None
    violations: int
    weight_stats: dict[str, tuple[float, float, float]]

    @property
    def response_rate(self) -> float:
        return self.responded / self.trials if self.trials else 0.0


@dataclass
class ImmediateRewardSetup:
    session: Session
    prompt_channel: FeatureInChannel
    output_channel: FeatureOutChannel
    rpe: ExternalRPE
    stdp: Hebbian
    rng: np.random.Generator
    edges: Mapping[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])
    reward_value: float = REWARD_VALUE
    ideal_duration: int = IDEAL_DURATION
    # Duration tolerance ramps from lenient (barely ever negative) to strict
    # over the first `tolerance_ramp_trials` trials, so early training can
    # first establish "respond at all" before being held to a precise
    # duration - see ramp().
    tolerance_start: float = 4.0
    tolerance_end: float = 1.0
    tolerance_ramp_trials: int = 150
    max_duration: int = 15
    timeout: int = 40
    cooldown: int = 25
    # Punishes output produced before the prompt arrives. Scaled up from 0
    # (no punishment) to full strength over the same kind of ramp, so a
    # network that hasn't learned to respond yet isn't also being punished
    # for every bit of exploratory noise.
    wait_min: int = 5
    wait_max: int = 20
    penalty_value: float = -REWARD_VALUE
    penalty_ramp_trials: int = 150

    trials: list[Trial] = field(default_factory=list[Trial])

    def weight_stats(self) -> dict[str, tuple[float, float, float]]:
        weight = self.session.snn.synapses.weight
        stats = {"all": (float(weight.min()), float(weight.mean()), float(weight.max()))}
        for name, indices in self.edges.items():
            edge_weight = weight[indices]
            stats[name] = (float(edge_weight.min()), float(edge_weight.mean()), float(edge_weight.max()))
        return stats


def run_trial(setup: ImmediateRewardSetup) -> Trial:
    trial_index = len(setup.trials)
    tolerance = ramp(trial_index, setup.tolerance_ramp_trials, setup.tolerance_start, setup.tolerance_end)
    penalty = setup.penalty_value * ramp(trial_index, setup.penalty_ramp_trials, 0.0, 1.0)

    violations = 0
    if setup.wait_max > 0:
        wait_ticks = int(setup.rng.integers(setup.wait_min, setup.wait_max + 1))
        for _ in range(wait_ticks):
            setup.session.tick()
            active = setup.output_channel.current is not None
            just_emitted = active and setup.output_channel.has_feature_changed
            if just_emitted:
                violations += 1
                if penalty != 0.0:
                    setup.rpe.notify(penalty)
                    setup.stdp.update(setup.session.snn)

    start_tick = setup.session.snn.tick
    setup.prompt_channel.write(PROMPT_FEATURE)

    response_tick: int | None = None
    for _ in range(setup.timeout):
        spikes = setup.session.tick()
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        if just_emitted:
            response_tick = spikes.tick
            break
    setup.prompt_channel.clear()

    duration: int | None = None
    reward_value: float | None = None
    rpe_value: float | None = None
    if response_tick is not None:
        duration = 1
        for _ in range(setup.max_duration - 1):
            setup.session.tick()
            if setup.output_channel.current is None:
                break
            duration += 1

        reward_value = duration_reward(duration, setup.ideal_duration, setup.reward_value, tolerance)
        setup.rpe.notify(reward_value)
        setup.stdp.update(setup.session.snn)
        rpe_value = setup.rpe.value

    for _ in range(setup.cooldown):
        setup.session.tick()

    trial = Trial(start_tick, setup.session.snn.tick, response_tick is not None, response_tick, duration, reward_value, rpe_value, violations)
    setup.trials.append(trial)
    return trial


def run_epoch(setup: ImmediateRewardSetup, ticks: int, index: int = 0) -> EpochStats:
    start_tick = setup.session.snn.tick
    start_trial_count = len(setup.trials)
    while setup.session.snn.tick - start_tick < ticks:
        run_trial(setup)

    epoch_trials = setup.trials[start_trial_count:]
    responded = [t for t in epoch_trials if t.responded]
    # -1: the earliest a response can appear is start_tick + 1 (SNN.step()
    # increments tick before returning), so 0 means "as early as possible".
    latencies = [t.response_tick - t.start_tick - 1 for t in responded if t.response_tick is not None]
    durations = [t.duration for t in responded if t.duration is not None]
    violations = sum(t.violations for t in epoch_trials)
    return EpochStats(
        index,
        start_tick,
        setup.session.snn.tick,
        len(epoch_trials),
        len(responded),
        (sum(latencies) / len(latencies)) if latencies else None,
        (sum(durations) / len(durations)) if durations else None,
        violations,
        setup.weight_stats(),
    )


def run_epochs(setup: ImmediateRewardSetup, ticks_per_epoch: int, epochs: int) -> list[EpochStats]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


def build_immediate_reward_experiment(
    seed: int = 0,
    *,
    prompt_amplitude: float = 0.7,
    prompt_duration: int = 4,
    timeout: int = 40,
    cooldown: int = 25,
    reward_value: float = REWARD_VALUE,
    tolerance_start: float = 4.0,
    tolerance_end: float = 1.0,
    tolerance_ramp_trials: int = 150,
    wait_min: int = 5,
    wait_max: int = 20,
    penalty_value: float = -REWARD_VALUE,
    penalty_ramp_trials: int = 150,
    # Low, unlike reward_shaping.py/turn_taking.py's shared 4, so burst
    # duration is governed by excitatory/inhibitory balance rather than
    # OUTPUT's own refractory period alone.
    refractory_ticks: int = 1,
    inhibitory: float = 0.3,
    # High for this network's current small size - inhibition's capacity
    # ceiling should come down as the network grows (see InhibitoryPlasticity).
    target_rate: float = 0.3,
) -> ImmediateRewardSetup:
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("PROMPT", (PROMPT_FEATURE,), 4),
            FeaturePopulationSpec("OUTPUT", ("SPEAK",), 6),
            NeuronPopulationSpec("HIDDEN", 64, inhibitory=inhibitory),
        ]
    )
    specs: list[ConnectionSpec] = []
    for source, target, fan_out, weight in (
        ("PROMPT", "HIDDEN", 16, DEFAULT_WEIGHT),
        ("HIDDEN", "HIDDEN", 8, RECURRENT_SEED_WEIGHT),
        ("HIDDEN", "OUTPUT", 8, DEFAULT_WEIGHT),
    ):
        target_count = layout.population(target).count
        specs.append(
            ConnectionSpec(source, target, FanOutSpec(min(fan_out, target_count)), weight)
        )
    connectivity = Connectivity.build(layout, specs, seed)
    snn = SNN.build(layout, connectivity)
    snn.neurons.refractory_ticks = refractory_ticks

    rng = np.random.default_rng(seed)

    hidden = layout.population("HIDDEN")

    prompt_channel = FeatureInChannel(
        layout.population("PROMPT"),
        PopulationEncoder(rng, prompt_amplitude),
        duration=prompt_duration,
    )
    output_channel = FeatureOutChannel(layout.population("OUTPUT"), PopulationDecoder(3.5))
    rpe = ExternalRPE()
    stdp = Hebbian(snn, third_factor=rpe)
    homeostatic = InhibitoryPlasticity(snn, target_rate=target_rate)
    background = BackgroundDrive([hidden], rng)

    session = Session(
        snn,
        pre=[prompt_channel, background],
        post=[output_channel, stdp, homeostatic],
    )

    return ImmediateRewardSetup(
        session,
        prompt_channel,
        output_channel,
        rpe,
        stdp,
        rng,
        edges=connectivity.edges,
        reward_value=reward_value,
        tolerance_start=tolerance_start,
        tolerance_end=tolerance_end,
        tolerance_ramp_trials=tolerance_ramp_trials,
        timeout=timeout,
        cooldown=cooldown,
        wait_min=wait_min,
        wait_max=wait_max,
        penalty_value=penalty_value,
        penalty_ramp_trials=penalty_ramp_trials,
    )


def _print_report(seed: int, stats: list[EpochStats]):
    print(f"\n[seed={seed}]")
    print(f"{'epoch':>5} {'trials':>6} {'resp%':>6} {'latency':>8} {'duration':>8} {'viol':>5} {'w(H->OUT)':>10} {'w(PROMPT->H)':>13}")
    for epoch in stats:
        latency = f"{epoch.mean_response_latency:.2f}" if epoch.mean_response_latency is not None else "-"
        duration = f"{epoch.mean_duration:.2f}" if epoch.mean_duration is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_OUTPUT", (0, 0, 0))[1]
        p_h = epoch.weight_stats.get("PROMPT_to_HIDDEN", (0, 0, 0))[1]
        print(
            f"{epoch.index:>5} {epoch.trials:>6} {epoch.response_rate * 100:>5.1f}% {latency:>8} {duration:>8} "
            f"{epoch.violations:>5} {h_out:>10.4f} {p_h:>13.4f}"
        )


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        setup = build_immediate_reward_experiment(seed)
        _print_report(seed, run_epochs(setup, 400, 15))
