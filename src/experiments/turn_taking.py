"""Serve-and-return experiment: drive the network turn-by-turn instead of
letting it free-run. Each turn presents a prompt, watches for a response
(from the moment the prompt is presented - not gated to a separate "wait"
sub-window, which would silently miss and fail to reward responses fast
enough to occur during the prompt itself), and if one occurs, rewards it by
saying a word ("yes"/"good") letter by letter and forcing the reward channel
up. If the network interrupts the word (cancel_on_interrupt=True), we say
"no" instead and force the reward channel to neutral (0.0).

Learning is intent-gated via ExternalRPE - see reward_shaping.py.
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
from ..network.connectivity import BernoulliTopologySpec, BimodalWeightSpec, ConnectionSpec, Connectivity
from ..network.population import FeaturePopulationSpec, NeuronPopulationSpec, NumericPopulationSpec, PopulationLayout
from ..network.snn import SNN
from ..session import Session
from .external_rpe import ExternalRPE

REWARD_WORDS: tuple[str, ...] = ("yes", "good")
NO_WORD: str = "no"
PROMPT_FEATURE = "CUE"


@dataclass
class RewardEvent:
    tick: int
    word: str
    cancelled: bool
    reward_value: float | None = None
    rpe: float | None = None


@dataclass
class Turn:
    start_tick: int
    end_tick: int
    responded: bool
    response_tick: int | None = None


@dataclass
class EpochStats:
    index: int
    start_tick: int
    end_tick: int
    turns: int
    responded: int
    mean_response_latency: float | None
    rewards_given: int
    rewards_cancelled: int
    weight_stats: dict[str, tuple[float, float, float]]

    @property
    def response_rate(self) -> float:
        return self.responded / self.turns if self.turns else 0.0


@dataclass
class TurnTakingSetup:
    session: Session
    prompt_channel: FeatureInChannel
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
    prompt_duration: int = 3
    response_timeout: int = 40
    # Long enough for HIDDEN to fully clear its refractory period before the
    # next prompt, so response reliability doesn't depend on where in that
    # recovery window the prompt happens to land.
    cooldown: int = 25

    turns: list[Turn] = field(default_factory=list[Turn])
    outputs: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    rewards_given: list[RewardEvent] = field(default_factory=list[RewardEvent])
    rewards_cancelled: list[RewardEvent] = field(default_factory=list[RewardEvent])

    def weight_stats(self) -> dict[str, tuple[float, float, float]]:
        weight = self.session.snn.synapses.weight
        stats = {"all": (float(weight.min()), float(weight.mean()), float(weight.max()))}
        for name, indices in self.edges.items():
            edge_weight = weight[indices]
            stats[name] = (float(edge_weight.min()), float(edge_weight.mean()), float(edge_weight.max()))
        return stats


def deliver_word(setup: TurnTakingSetup, word: str) -> tuple[str, bool]:
    """Feed `word` letter by letter. If interrupted and cancel_on_interrupt is
    set, switch immediately to NO_WORD instead - itself not cancellable.
    Returns the word actually delivered and whether it was cancelled."""
    for character in word:
        setup.word_channel.write(character)

    remaining = len(word)
    cancelled = False
    while remaining > 0:
        setup.session.tick()
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        remaining -= 1
        if just_emitted and setup.cancel_on_interrupt and not cancelled:
            setup.word_channel.clear()
            cancelled = True
            word = NO_WORD
            for character in NO_WORD:
                setup.word_channel.write(character)
            remaining = len(NO_WORD)
    return word, cancelled


def run_turn(setup: TurnTakingSetup) -> Turn:
    start_tick = setup.session.snn.tick
    setup.prompt_channel.write(PROMPT_FEATURE)

    response_tick = None
    for _ in range(setup.prompt_duration + setup.response_timeout):
        spikes = setup.session.tick()
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        if just_emitted:
            response_tick = spikes.tick
            break
    setup.prompt_channel.clear()

    if response_tick is not None:
        word = str(setup.rng.choice(setup.reward_words))
        delivered_word, cancelled = deliver_word(setup, word)

        value = 0.0 if cancelled else setup.reward_spike
        setup.reward_channel.force(value)
        setup.rpe.notify(value)
        if not cancelled:
            setup.stdp.update(setup.session.snn)

        event = RewardEvent(setup.session.snn.tick, delivered_word, cancelled, value, setup.rpe.last_rpe)
        (setup.rewards_cancelled if cancelled else setup.rewards_given).append(event)

    for _ in range(setup.cooldown):
        setup.session.tick()

    turn = Turn(start_tick, setup.session.snn.tick, response_tick is not None, response_tick)
    setup.turns.append(turn)
    return turn


def run(setup: TurnTakingSetup, ticks: int):
    end_tick = setup.session.snn.tick + ticks
    while setup.session.snn.tick < end_tick:
        run_turn(setup)
    return setup


def run_epoch(setup: TurnTakingSetup, ticks: int, index: int = 0) -> EpochStats:
    start_tick = setup.session.snn.tick
    start_turn_count = len(setup.turns)
    start_given, start_cancelled = len(setup.rewards_given), len(setup.rewards_cancelled)

    run(setup, ticks)

    epoch_turns = setup.turns[start_turn_count:]
    responded = [t for t in epoch_turns if t.responded]
    latencies = [t.response_tick - t.start_tick for t in responded if t.response_tick is not None]
    return EpochStats(
        index,
        start_tick,
        setup.session.snn.tick,
        len(epoch_turns),
        len(responded),
        (sum(latencies) / len(latencies)) if latencies else None,
        len(setup.rewards_given) - start_given,
        len(setup.rewards_cancelled) - start_cancelled,
        setup.weight_stats(),
    )


def run_epochs(setup: TurnTakingSetup, ticks_per_epoch: int, epochs: int) -> list[EpochStats]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


DEFAULT_WEIGHT = BimodalWeightSpec(0.6, -0.1, 0.15, 0.04, 0.02)
REWARD_SEED_WEIGHT = BimodalWeightSpec(0.08, -0.03, 0.15, 0.01, 0.005)
RECURRENT_SEED_WEIGHT = BimodalWeightSpec(0.08, -0.03, 0.15, 0.01, 0.005)


def build_turn_taking_experiment(
    seed: int = 0,
    *,
    cancel_on_interrupt: bool,
    prompt_amplitude: float = 5.0,
    prompt_duration: int = 3,
    # Shared with immediate_reward.py/reward_shaping.py for a common
    # baseline.
    background_amplitude: float = 0.12,
    refractory_ticks: int = 4,
) -> TurnTakingSetup:
    letters = tuple(sorted(set("".join(REWARD_WORDS) + NO_WORD)))
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("PROMPT", (PROMPT_FEATURE,), 4),
            FeaturePopulationSpec("REWARD_WORD", letters, 4),
            FeaturePopulationSpec("OUTPUT", ("SPEAK",), 6),
            NeuronPopulationSpec("HIDDEN", 64),
            NumericPopulationSpec("REWARD", 8, 8),
        ]
    )
    specs: list[ConnectionSpec] = []
    for source, target, fan_out, weight in (
        ("PROMPT", "HIDDEN", 16, DEFAULT_WEIGHT),
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

    rng = np.random.default_rng(seed)

    # Randomized per-tick encoding (see PopulationEncoder.rng) spreads
    # activation across the presentation instead of the 4 PROMPT neurons
    # firing in lockstep on tick one.
    prompt_channel = FeatureInChannel(
        layout.population("PROMPT"),
        PopulationEncoder(prompt_amplitude, rng=rng),
        duration=prompt_duration,
    )
    word_channel = FeatureInChannel(layout.population("REWARD_WORD"), PopulationEncoder(4.0))
    output_channel = FeatureOutChannel(layout.population("OUTPUT"), PopulationDecoder(3.5))
    reward_channel = NumericChannel(layout.population("REWARD"), rng)
    rpe = ExternalRPE()
    stdp = STDP(snn, third_factor=rpe)
    background = BackgroundDrive([layout.population("HIDDEN")], background_amplitude, rng)

    session = Session(
        snn,
        pre=[prompt_channel, word_channel, reward_channel, background],
        post=[output_channel, reward_channel, stdp],
    )

    return TurnTakingSetup(
        session,
        prompt_channel,
        word_channel,
        output_channel,
        reward_channel,
        rpe,
        stdp,
        rng,
        cancel_on_interrupt,
        edges=connectivity.edges,
        prompt_duration=prompt_duration,
    )


def _print_report(seed: int, label: str, stats: list[EpochStats]):
    print(f"\n[seed={seed} {label}]")
    print(f"{'epoch':>5} {'turns':>6} {'resp%':>6} {'latency':>8} {'given':>6} {'cancelled':>10} {'w(H->OUT)':>10} {'w(H->H)':>9}")
    for epoch in stats:
        latency = f"{epoch.mean_response_latency:.2f}" if epoch.mean_response_latency is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_OUTPUT", (0, 0, 0))[1]
        h_h = epoch.weight_stats.get("HIDDEN_to_HIDDEN", (0, 0, 0))[1]
        print(
            f"{epoch.index:>5} {epoch.turns:>6} {epoch.response_rate * 100:>5.1f}% {latency:>8} "
            f"{epoch.rewards_given:>6} {epoch.rewards_cancelled:>10} {h_out:>10.4f} {h_h:>9.4f}"
        )


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        for cancel_on_interrupt in (True, False):
            label = "cancel-on-interrupt" if cancel_on_interrupt else "ignore-interrupt"
            setup = build_turn_taking_experiment(seed, cancel_on_interrupt=cancel_on_interrupt)
            _print_report(seed, label, run_epochs(setup, 500, 10))
