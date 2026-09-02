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
from ..interaction.stdp import STDP
from ..network.connectivity import BernoulliTopologySpec, BimodalWeightSpec, ConnectionSpec, Connectivity
from ..network.population import FeaturePopulationSpec, NeuronPopulationSpec, PopulationLayout
from ..network.snn import SNN
from ..session import Session
from .external_rpe import ExternalRPE

PROMPT_FEATURE = "CUE"
REWARD_VALUE = 0.75
DEFAULT_WEIGHT = BimodalWeightSpec(0.6, -0.1, 0.15, 0.04, 0.02)
# Seeded much weaker than DEFAULT_WEIGHT so the network can still fall quiet
# between responses instead of self-sustaining - see reward_shaping.py.
RECURRENT_SEED_WEIGHT = BimodalWeightSpec(0.08, -0.03, 0.15, 0.01, 0.005)


@dataclass
class Trial:
    start_tick: int
    end_tick: int
    responded: bool
    response_tick: int | None = None
    reward_value: float | None = None
    rpe: float | None = None


@dataclass
class EpochStats:
    index: int
    start_tick: int
    end_tick: int
    trials: int
    responded: int
    mean_response_latency: float | None
    weight_stats: dict[str, tuple[float, float, float]]

    @property
    def response_rate(self) -> float:
        return self.responded / self.trials if self.trials else 0.0


@dataclass
class ImmediateRewardSetup:
    """Wiring for the experiment plus accumulated trial history - the trial
    logic itself lives in run_trial()/run_epoch(), not on this object."""

    session: Session
    prompt_channel: FeatureInChannel
    output_channel: FeatureOutChannel
    rpe: ExternalRPE
    stdp: STDP
    edges: Mapping[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])
    reward_value: float = REWARD_VALUE
    timeout: int = 40
    cooldown: int = 25

    trials: list[Trial] = field(default_factory=list[Trial])

    def weight_stats(self) -> dict[str, tuple[float, float, float]]:
        weight = self.session.snn.synapses.weight
        stats = {"all": (float(weight.min()), float(weight.mean()), float(weight.max()))}
        for name, indices in self.edges.items():
            edge_weight = weight[indices]
            stats[name] = (float(edge_weight.min()), float(edge_weight.mean()), float(edge_weight.max()))
        return stats


def run_trial(setup: ImmediateRewardSetup) -> Trial:
    """Present the prompt and run the session until a response is detected or
    the timeout elapses, rewarding immediately (zero delay) if one occurs.
    """
    start_tick = setup.session.snn.tick
    setup.prompt_channel.write(PROMPT_FEATURE)

    response_tick: int | None = None
    reward_value: float | None = None
    rpe_value: float | None = None
    for _ in range(setup.timeout):
        spikes = setup.session.tick()
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        if just_emitted:
            response_tick = spikes.tick
            setup.rpe.notify(setup.reward_value)
            setup.stdp.update(setup.session.snn)
            reward_value = setup.reward_value
            rpe_value = setup.rpe.last_rpe
            break
    setup.prompt_channel.clear()

    # A quiet gap so refractory states clear before the next prompt, rather
    # than immediately overlapping the tail of this trial.
    for _ in range(setup.cooldown):
        setup.session.tick()

    trial = Trial(start_tick, setup.session.snn.tick, response_tick is not None, response_tick, reward_value, rpe_value)
    setup.trials.append(trial)
    return trial


def run_epoch(setup: ImmediateRewardSetup, ticks: int, index: int = 0) -> EpochStats:
    """Run whole trials until at least `ticks` have elapsed since the start of
    this epoch. The boundary is checked between trials against the session's
    own tick count, never mid-trial, so trial length can vary freely without
    ever needing to split one across an epoch boundary.
    """
    start_tick = setup.session.snn.tick
    start_trial_count = len(setup.trials)
    while setup.session.snn.tick - start_tick < ticks:
        run_trial(setup)

    epoch_trials = setup.trials[start_trial_count:]
    responded = [t for t in epoch_trials if t.responded]
    latencies = [t.response_tick - t.start_tick for t in responded if t.response_tick is not None]
    return EpochStats(
        index,
        start_tick,
        setup.session.snn.tick,
        len(epoch_trials),
        len(responded),
        (sum(latencies) / len(latencies)) if latencies else None,
        setup.weight_stats(),
    )


def run_epochs(setup: ImmediateRewardSetup, ticks_per_epoch: int, epochs: int) -> list[EpochStats]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


def build_immediate_reward_experiment(
    seed: int = 0,
    *,
    prompt_amplitude: float = 0.7,
    prompt_duration: int = 4,
    # Shared with reward_shaping.py/turn_taking.py for a common baseline -
    # kept at the low end of that range because, combined with HIDDEN ->
    # HIDDEN recurrence, anything higher lets the network produce output
    # from ambient excitation alone regardless of the prompt - self-defeating
    # for an experiment meant to test whether reward reinforces a
    # prompt-driven response specifically.
    background_amplitude: float = 0.12,
    timeout: int = 40,
    cooldown: int = 25,
    reward_value: float = REWARD_VALUE,
    refractory_ticks: int = 4,
) -> ImmediateRewardSetup:
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("PROMPT", (PROMPT_FEATURE,), 4),
            FeaturePopulationSpec("OUTPUT", ("SPEAK",), 6),
            NeuronPopulationSpec("HIDDEN", 64),
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
            ConnectionSpec(source, target, BernoulliTopologySpec(min(fan_out, target_count)), weight)
        )
    connectivity = Connectivity.build(layout, specs, seed)
    snn = SNN.build(layout, connectivity)
    snn.neurons.refractory_ticks = refractory_ticks

    rng = np.random.default_rng(seed)

    hidden = layout.population("HIDDEN")

    # Randomized per-tick encoding spreads PROMPT's 4 neurons' firing across
    # the presentation instead of a synchronized lockstep instant, which also
    # supplies the trial-to-trial variability needed to bootstrap learning.
    prompt_channel = FeatureInChannel(
        layout.population("PROMPT"),
        PopulationEncoder(prompt_amplitude, rng=rng),
        duration=prompt_duration,
    )
    output_channel = FeatureOutChannel(layout.population("OUTPUT"), PopulationDecoder(3.5))
    rpe = ExternalRPE()
    stdp = STDP(snn, third_factor=rpe)
    background = BackgroundDrive([hidden], background_amplitude, rng)

    session = Session(
        snn,
        pre=[prompt_channel, background],
        post=[output_channel, stdp],
    )

    return ImmediateRewardSetup(
        session,
        prompt_channel,
        output_channel,
        rpe,
        stdp,
        edges=connectivity.edges,
        reward_value=reward_value,
        timeout=timeout,
        cooldown=cooldown,
    )


def _print_report(seed: int, stats: list[EpochStats]):
    print(f"\n[seed={seed}]")
    print(f"{'epoch':>5} {'trials':>6} {'resp%':>6} {'latency':>8} {'w(H->OUT)':>10} {'w(PROMPT->H)':>13}")
    for epoch in stats:
        latency = f"{epoch.mean_response_latency:.2f}" if epoch.mean_response_latency is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_OUTPUT", (0, 0, 0))[1]
        p_h = epoch.weight_stats.get("PROMPT_to_HIDDEN", (0, 0, 0))[1]
        print(
            f"{epoch.index:>5} {epoch.trials:>6} {epoch.response_rate * 100:>5.1f}% {latency:>8} "
            f"{h_out:>10.4f} {p_h:>13.4f}"
        )


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        setup = build_immediate_reward_experiment(seed)
        _print_report(seed, run_epochs(setup, 400, 15))
