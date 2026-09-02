"""Reward-shaping experiment: reward the network whenever it moves from
silence to making output, by saying a short word ("yes"/"good") letter by
letter and forcing the reward channel up once it finishes. If the network
interrupts the word (cancel_on_interrupt=True), we say "no" instead and force
the reward channel to neutral (0.0) - a genuine, learnable consequence for
interrupting rather than silently dropping the reward.

Learning is intent-gated via ExternalRPE: the neural REWARD population and
NumericChannel stay wired up and driven by force(), but the STDP update uses
the value we know we delivered rather than reading it back from spikes.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..interaction.background import BackgroundDrive
from ..interaction.channels import FeatureInChannel, FeatureOutChannel, NumericChannel, PopulationDecoder, PopulationEncoder
from ..interaction.stdp import STDP
from ..network.connectivity import BernoulliTopologySpec, WeightSpec, ConnectionSpec, Connectivity
from ..network.population import FeaturePopulationSpec, NeuronPopulationSpec, NumericPopulationSpec, PopulationLayout
from ..network.snn import SNN
from ..session import Session
from .external_rpe import ExternalRPE

REWARD_WORDS: tuple[str, ...] = ("yes", "good")
NO_WORD: str = "no"


@dataclass
class RewardEvent:
    tick: int
    word: str
    cancelled: bool
    reward_value: float | None = None
    rpe: float | None = None


@dataclass
class EpochStats:
    index: int
    start_tick: int
    end_tick: int
    active_ticks: int
    silent_ticks: int
    bursts: int
    mean_silence_run: float | None
    rewards_given: int
    rewards_cancelled: int
    weight_stats: dict[str, tuple[float, float, float]]

    @property
    def active_ratio(self) -> float:
        total = self.active_ticks + self.silent_ticks
        return self.active_ticks / total if total else 0.0


@dataclass
class RewardSetup:
    session: Session
    word_channel: FeatureInChannel
    output_channel: FeatureOutChannel
    reward_channel: NumericChannel
    rpe: ExternalRPE
    stdp: STDP
    rng: np.random.Generator
    cancel_on_interrupt: bool
    edges: Mapping[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])
    reward_words: tuple[str, ...] = REWARD_WORDS
    reward_spike: float = 0.75
    # Cancellation is a real, punished consequence, not just a withheld
    # reward - see immediate_reward.py's duration_reward for why an
    # actual negative RPE (LTD) is needed rather than 0.0 (no-op).
    cancel_penalty: float = -0.75

    outputs: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    silent_ticks: int = 0
    active_ticks: int = 0
    silence_runs: list[int] = field(default_factory=list[int])
    rewards_given: list[RewardEvent] = field(default_factory=list[RewardEvent])
    rewards_cancelled: list[RewardEvent] = field(default_factory=list[RewardEvent])

    _was_silent: bool = True
    _current_silence_run: int = 0

    def weight_stats(self) -> dict[str, tuple[float, float, float]]:
        weight = self.session.snn.synapses.weight
        stats = {"all": (float(weight.min()), float(weight.mean()), float(weight.max()))}
        for name, indices in self.edges.items():
            edge_weight = weight[indices]
            stats[name] = (float(edge_weight.min()), float(edge_weight.mean()), float(edge_weight.max()))
        return stats


def observe(setup: RewardSetup):
    """Advance one tick, updating active/silent/output bookkeeping. This is
    the single place ticks are consumed from, so that ticks spent inside
    deliver_word() are tracked identically to ticks spent free-running -
    silence detection depends on every tick being accounted for the same way
    regardless of which loop is currently driving the session."""
    spikes = setup.session.tick()
    active = setup.output_channel.current is not None
    just_emitted = active and setup.output_channel.has_feature_changed

    if active:
        setup.active_ticks += 1
        if just_emitted and setup.output_channel.current is not None:
            setup.outputs.append((spikes.tick, setup.output_channel.current.feature))
    else:
        setup.silent_ticks += 1
        setup._current_silence_run += 1

    was_silent = setup._was_silent
    setup._was_silent = not active
    return spikes, just_emitted, was_silent


def deliver_word(setup: RewardSetup, word: str) -> tuple[str, bool]:
    """Feed `word` letter by letter. If interrupted and cancel_on_interrupt is
    set, switch immediately to NO_WORD instead - itself not cancellable.
    Returns the word actually delivered and whether it was cancelled."""
    for character in word:
        setup.word_channel.write(character)

    remaining = len(word)
    cancelled = False
    while remaining > 0:
        _, just_emitted, _ = observe(setup)
        remaining -= 1
        if just_emitted and setup.cancel_on_interrupt and not cancelled:
            setup.word_channel.clear()
            cancelled = True
            word = NO_WORD
            for character in NO_WORD:
                setup.word_channel.write(character)
            remaining = len(NO_WORD)
    return word, cancelled


def deliver_reward(setup: RewardSetup, tick: int):
    word = str(setup.rng.choice(setup.reward_words))
    delivered_word, cancelled = deliver_word(setup, word)

    value = setup.cancel_penalty if cancelled else setup.reward_spike
    setup.reward_channel.force(value)
    setup.rpe.notify(value)
    setup.stdp.update(setup.session.snn)

    event = RewardEvent(tick, delivered_word, cancelled, value, setup.rpe.last_rpe)
    (setup.rewards_cancelled if cancelled else setup.rewards_given).append(event)


def run(setup: RewardSetup, ticks: int):
    """Free-run for `ticks` ticks, rewarding (or withholding) every time the
    network moves from silence to making output."""
    end_tick = setup.session.snn.tick + ticks
    while setup.session.snn.tick < end_tick:
        spikes, just_emitted, was_silent = observe(setup)
        if just_emitted and was_silent:
            setup.silence_runs.append(setup._current_silence_run)
            setup._current_silence_run = 0
            deliver_reward(setup, spikes.tick)
    return setup


def run_epoch(setup: RewardSetup, ticks: int, index: int = 0) -> EpochStats:
    start_tick = setup.session.snn.tick
    start_active, start_silent = setup.active_ticks, setup.silent_ticks
    start_given, start_cancelled = len(setup.rewards_given), len(setup.rewards_cancelled)
    start_bursts, start_runs = len(setup.outputs), len(setup.silence_runs)

    run(setup, ticks)

    epoch_runs = setup.silence_runs[start_runs:]
    return EpochStats(
        index,
        start_tick,
        setup.session.snn.tick,
        setup.active_ticks - start_active,
        setup.silent_ticks - start_silent,
        len(setup.outputs) - start_bursts,
        (sum(epoch_runs) / len(epoch_runs)) if epoch_runs else None,
        len(setup.rewards_given) - start_given,
        len(setup.rewards_cancelled) - start_cancelled,
        setup.weight_stats(),
    )


def run_epochs(setup: RewardSetup, ticks_per_epoch: int, epochs: int) -> list[EpochStats]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


DEFAULT_WEIGHT = WeightSpec(0.6, 0.04)
# HIDDEN -> REWARD and HIDDEN -> HIDDEN are seeded much weaker than the other
# pathways: the former so ambient reward-population activity starts small
# rather than swamping the ExternalRPE signal, the latter so the network can
# actually fall silent between bursts instead of self-sustaining forever.
REWARD_SEED_WEIGHT = WeightSpec(0.08, 0.01)
RECURRENT_SEED_WEIGHT = WeightSpec(0.08, 0.01)


def build_reward_experiment(
    seed: int = 0,
    *,
    cancel_on_interrupt: bool,
    hot_voltage: float = 0.3,
    # Shared with immediate_reward.py/turn_taking.py for a common baseline.
    # At this level reward_shaping currently goes silent (it has no external
    # prompt to kickstart activity the way the other two do) - a known,
    # separately-tracked issue, not something to retune around yet.
    background_amplitude: float = 0.12,
    refractory_ticks: int = 4,
) -> RewardSetup:
    letters = tuple(sorted(set("".join(REWARD_WORDS) + NO_WORD)))
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("REWARD_WORD", letters, 4),
            FeaturePopulationSpec("OUTPUT", ("SPEAK",), 6),
            NeuronPopulationSpec("HIDDEN", 64, inhibitory=0.3),
            NumericPopulationSpec("REWARD", 8, 8),
        ]
    )
    specs: list[ConnectionSpec] = []
    for source, target, fan_out, weight in (
        ("REWARD_WORD", "HIDDEN", 16, DEFAULT_WEIGHT),
        ("HIDDEN", "HIDDEN", 8, RECURRENT_SEED_WEIGHT),
        ("HIDDEN", "OUTPUT", 8, DEFAULT_WEIGHT),
        ("HIDDEN", "REWARD", 8, REWARD_SEED_WEIGHT),
    ):
        target_count = layout.population(target).count
        specs.append(
            ConnectionSpec(source, target, BernoulliTopologySpec(min(fan_out, target_count)), weight)
        )
    connectivity = Connectivity.build(layout, specs, seed)
    snn = SNN.build(layout, connectivity)
    snn.neurons.refractory_ticks = refractory_ticks

    hidden = layout.population("HIDDEN")
    snn.neurons.voltage[hidden.bounds] = hot_voltage

    rng = np.random.default_rng(seed)

    word_channel = FeatureInChannel(layout.population("REWARD_WORD"), PopulationEncoder(4.0))
    output_channel = FeatureOutChannel(layout.population("OUTPUT"), PopulationDecoder(3.5))
    reward_channel = NumericChannel(layout.population("REWARD"), rng)
    rpe = ExternalRPE()
    stdp = STDP(snn, third_factor=rpe)
    background = BackgroundDrive([hidden], background_amplitude, rng)

    session = Session(
        snn,
        pre=[word_channel, reward_channel, background],
        post=[output_channel, reward_channel, stdp],
    )

    return RewardSetup(
        session,
        word_channel,
        output_channel,
        reward_channel,
        rpe,
        stdp,
        rng,
        cancel_on_interrupt,
        edges=connectivity.edges,
    )


def _print_report(seed: int, label: str, stats: list[EpochStats]):
    print(f"\n[seed={seed} {label}]")
    print(f"{'epoch':>5} {'active%':>8} {'mean_silence':>13} {'given':>6} {'cancelled':>10} {'w(H->OUT)':>10} {'w(H->H)':>9}")
    for epoch in stats:
        mean_silence = f"{epoch.mean_silence_run:.2f}" if epoch.mean_silence_run is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_OUTPUT", (0, 0, 0))[1]
        h_h = epoch.weight_stats.get("HIDDEN_to_HIDDEN", (0, 0, 0))[1]
        print(
            f"{epoch.index:>5} {epoch.active_ratio * 100:>7.1f}% {mean_silence:>13} "
            f"{epoch.rewards_given:>6} {epoch.rewards_cancelled:>10} {h_out:>10.4f} {h_h:>9.4f}"
        )


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        for cancel_on_interrupt in (True, False):
            label = "cancel-on-interrupt" if cancel_on_interrupt else "ignore-interrupt"
            setup = build_reward_experiment(seed, cancel_on_interrupt=cancel_on_interrupt)
            _print_report(seed, label, run_epochs(setup, 200, 10))
