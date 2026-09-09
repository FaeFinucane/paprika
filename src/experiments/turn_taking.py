"""Serve-and-return experiment: drive the network turn-by-turn instead of
letting it free-run. Each turn presents a prompt, watches for a response
(from the moment the prompt is presented - not gated to a separate "wait"
sub-window, which would silently miss and fail to reward responses fast
enough to occur during the prompt itself), and if one occurs, rewards it by
saying a word ("yes"/"good") letter by letter and forcing the reward channel
up. If the network interrupts the word (cancel_on_interrupt=True), we say
"no" instead and force the reward channel to neutral (0.0).

Learning is TD-style via two neural populations: REWARD (forced with ground
truth - the "did this actually happen" signal) and PREDICTOR (never forced,
trained purely from the rpe it helps produce - the network's own value
prediction).
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..interaction.background import BackgroundDrive
from ..interaction.channels import FeatureInChannel, FeatureOutChannel, NumericChannel, PopulationDecoder, PopulationEncoder
from ..interaction.plasticity import RPE, Hebbian, InhibitoryPlasticity
from ..network.connectivity import FanInSpec, FanOutSpec, WeightSpec, ConnectionSpec, build_synapses
from ..network.population import FeaturePopulationSpec, NeuronPopulationSpec, NumericPopulationSpec, PopulationLayout
from ..network.snn import SNN
from ..session import Session
from .curriculum import duration_reward, ramp
from .diagnostics import Cue, EpochStats, PredictorReport, Trial, TrialRecorder, attach_predictor_diagnostic, predictor_report, summarize_epoch

REWARD_WORDS: tuple[str, ...] = ("yes", "good")
NO_WORD: str = "no"
PROMPT_FEATURE = "CUE"


@dataclass
class TurnTakingSetup:
    session: Session
    prompt_channel: FeatureInChannel
    word_channel: FeatureInChannel
    output_channel: FeatureOutChannel
    reward_channel: NumericChannel
    predictor_channel: NumericChannel
    stdp: Hebbian
    rng: np.random.Generator
    cancel_on_interrupt: bool
    trial_recorder: TrialRecorder
    specs: Sequence[ConnectionSpec] = field(default_factory=list)
    reward_words: tuple[str, ...] = REWARD_WORDS
    reward_spike: float = 1.0
    ideal_duration: int = 3
    # Ramps lenient -> strict over trials, like penalty_ramp_rewards below -
    # a short first burst (duration=1) shouldn't be punished as harshly as a
    # network that's already learned to respond at all.
    tolerance_start: float = 4.0
    tolerance_end: float = 1.0
    tolerance_ramp_turns: int = 150
    max_duration: int = 15
    cancel_penalty: float = -0.75
    # Cancellation punishment scales with recent response reliability
    # (recent_response_rate) and with how many rewards have actually been
    # given so far (penalty_ramp_rewards) - not punished until the network
    # has both learned to respond and actually earned some rewards.
    competence_window: int = 20
    penalty_ramp_rewards: int = 10
    # Caps how many times "no" gets repeated if the network keeps talking
    # over it, so a network that never stops interrupting can't loop forever.
    max_punishment_repeats: int = 3
    # Unprompted output before CUE arrives is punished the same way.
    wait_min: int = 5
    wait_max: int = 20
    prompt_duration: int = 3
    response_timeout: int = 40
    cooldown: int = 25

    def weight_stats(self) -> dict[str, tuple[float, float, float]]:
        synapses = self.session.snn.synapses
        weight = synapses.weight
        stats = {"all": (float(weight.min()), float(weight.mean()), float(weight.max()))}
        layout = self.session.snn.layout
        for spec in self.specs:
            source, target = layout.population(spec.source).bounds, layout.population(spec.target).bounds
            mask = (
                (synapses.source >= source.start) & (synapses.source < source.stop)
                & (synapses.target >= target.start) & (synapses.target < target.stop)
            )
            edge_weight = weight[mask]
            stats[spec.name] = (float(edge_weight.min()), float(edge_weight.mean()), float(edge_weight.max()))
        return stats


def total_reward_count(setup: TurnTakingSetup) -> int:
    return sum(t.count_cue(Cue.REWARD) for t in setup.trial_recorder.trials)


def recent_response_rate(setup: TurnTakingSetup) -> float:
    recent = setup.trial_recorder.trials[-setup.competence_window:]
    if not recent:
        return 0.0
    return sum(t.has_cue(Cue.RESPONSE) for t in recent) / len(recent)


def start_trial(setup: TurnTakingSetup):
    setup.trial_recorder.start(setup.session.snn.tick)


def mark(setup: TurnTakingSetup, cue: Cue):
    setup.trial_recorder.mark(cue, setup.session.snn.tick)


def finalize_trial(setup: TurnTakingSetup) -> Trial:
    return setup.trial_recorder.finalize(setup.session.snn.tick)


def step(setup: TurnTakingSetup):
    """Advance one tick and apply the continuous three-factor STDP update -
    the internal RPE (a TD-style diff of the REWARD channel) needs sampling
    every tick, not just at reward-delivery moments, to actually function as
    a value predictor rather than a one-shot associative signal."""
    spikes = setup.session.tick()
    setup.stdp.update(setup.session.snn)
    return spikes


def deliver_no(setup: TurnTakingSetup) -> bool:
    """Say 'no' letter by letter, delivering competence-scaled punishment on
    the last letter. Returns whether output occurred during delivery - the
    caller repeats this if so, since talking over the punishment shouldn't
    be a way to dodge it."""
    for character in NO_WORD:
        setup.word_channel.write(character)

    remaining = len(NO_WORD)
    interrupted = False
    while remaining > 0:
        step(setup)
        active = setup.output_channel.current is not None
        if active and setup.output_channel.has_feature_changed:
            interrupted = True
        remaining -= 1

    competence = recent_response_rate(setup) * ramp(total_reward_count(setup), setup.penalty_ramp_rewards, 0.0, 1.0)
    value = setup.cancel_penalty * competence
    setup.reward_channel.force(value, duration=3)
    step(setup)
    mark(setup, Cue.PUNISH)
    setup.trial_recorder.set_extra("punish_value", value)
    return interrupted


def punish(setup: TurnTakingSetup):
    interrupted = True
    repeats = 0
    while interrupted and repeats < setup.max_punishment_repeats:
        interrupted = deliver_no(setup)
        repeats += 1


def deliver_word(setup: TurnTakingSetup, word: str) -> tuple[str, bool]:
    """Feed `word` letter by letter. If interrupted and cancel_on_interrupt
    is set, switch to punish() instead - itself repeated if interrupted too.
    Returns the word actually delivered and whether it was cancelled."""
    for character in word:
        setup.word_channel.write(character)

    remaining = len(word)
    cancelled = False
    while remaining > 0:
        step(setup)
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        remaining -= 1
        if just_emitted and setup.cancel_on_interrupt and not cancelled:
            setup.word_channel.clear()
            cancelled = True

    if cancelled:
        punish(setup)
        return NO_WORD, True
    return word, False


def run_turn(setup: TurnTakingSetup) -> Trial:
    start_trial(setup)

    if setup.wait_max > 0:
        wait_ticks = int(setup.rng.integers(setup.wait_min, setup.wait_max + 1))
        for _ in range(wait_ticks):
            step(setup)
            active = setup.output_channel.current is not None
            if active and setup.output_channel.has_feature_changed:
                punish(setup)
                break
    mark(setup, Cue.BASELINE)

    setup.prompt_channel.write(PROMPT_FEATURE)

    responded = False
    for i in range(setup.prompt_duration + setup.response_timeout):
        step(setup)
        if i == 0:
            # After the cue has had one tick to actually reach HIDDEN and
            # propagate back, not at the instant of writing it - marking
            # immediately after write() would be identical to baseline by
            # construction, since no simulation time would have passed.
            mark(setup, Cue.PROMPT)
        active = setup.output_channel.current is not None
        just_emitted = active and setup.output_channel.has_feature_changed
        if just_emitted:
            responded = True
            break
    setup.prompt_channel.clear()

    if responded:
        mark(setup, Cue.RESPONSE)
        # Reward isn't delivered until output actually stops - a real
        # consequence of the delay is that "stop outputting" becomes a
        # precondition for any reward at all, not something we reward
        # directly.
        duration = 1
        for _ in range(setup.max_duration - 1):
            step(setup)
            if setup.output_channel.current is None:
                break
            duration += 1
        setup.trial_recorder.set_extra("response_duration", duration)

        word = str(setup.rng.choice(setup.reward_words))
        delivered_word, cancelled = deliver_word(setup, word)

        if not cancelled:
            tolerance = ramp(len(setup.trial_recorder.trials), setup.tolerance_ramp_turns, setup.tolerance_start, setup.tolerance_end)
            value = duration_reward(duration, setup.ideal_duration, setup.reward_spike, tolerance)
            setup.reward_channel.force(value, duration=3)
            step(setup)
            mark(setup, Cue.REWARD)
            setup.trial_recorder.set_extra("reward_value", value)
            setup.trial_recorder.set_extra("delivered_word", delivered_word)
        else:
            mark(setup, Cue.CANCELLED)

    for _ in range(setup.cooldown):
        step(setup)

    return finalize_trial(setup)


def run(setup: TurnTakingSetup, ticks: int):
    end_tick = setup.session.snn.tick + ticks
    while setup.session.snn.tick < end_tick:
        run_turn(setup)
    return setup


def run_epoch(setup: TurnTakingSetup, ticks: int, index: int = 0) -> tuple[EpochStats, PredictorReport]:
    start_tick = setup.session.snn.tick
    start_trial_count = len(setup.trial_recorder.trials)

    run(setup, ticks)

    epoch_trials = setup.trial_recorder.trials[start_trial_count:]
    stats = summarize_epoch(epoch_trials, index, start_tick, setup.session.snn.tick, setup.weight_stats())
    discount = getattr(setup.stdp.third_factor, "discount", 0.95)
    return stats, predictor_report(epoch_trials, discount)


def run_epochs(setup: TurnTakingSetup, ticks_per_epoch: int, epochs: int) -> list[tuple[EpochStats, PredictorReport]]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


DEFAULT_WEIGHT = WeightSpec(0.6, 0.04)
# Stronger than DEFAULT_WEIGHT to compensate for HIDDEN->OUTPUT's sparser
# fan-in (see FanOutSpec/FanInSpec choice below) still reliably triggering
# OUTPUT.
OUTPUT_SEED_WEIGHT = WeightSpec(0.9, 0.04)
REWARD_SEED_WEIGHT = WeightSpec(0.08, 0.01)
PREDICTOR_SEED_WEIGHT = WeightSpec(0.08, 0.01)
RECURRENT_SEED_WEIGHT = WeightSpec(0.08, 0.01)


def build_turn_taking_experiment(
    seed: int = 0,
    *,
    cancel_on_interrupt: bool,
    amplitude: float = 1.0,
    prompt_duration: int = 3,
    refractory_ticks: int = 1,
    learning_rate: float = 0.001,
    # Measured HIDDEN mean per-tick firing probability during normal
    # (prompted) operation - see the homeostatic plasticity design.
    target_rate: float = 0.005,
    # PREDICTOR's seed weight alone left it too weak for HIDDEN to reliably
    # drive at all - renormalized total, measured empirically (see
    # conversation), same reasoning as discrimination.py's fix.
    predictor_renormalize_total: float = 5.0,
) -> TurnTakingSetup:
    letters = tuple(sorted(set("".join(REWARD_WORDS) + NO_WORD)))
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("PROMPT", (PROMPT_FEATURE,), 4),
            FeaturePopulationSpec("REWARD_WORD", letters, 4),
            FeaturePopulationSpec("OUTPUT", ("SPEAK",), 6),
            NeuronPopulationSpec("HIDDEN", 64, inhibitory=0.3),
            NumericPopulationSpec("REWARD", 8, 8),
            NumericPopulationSpec("PREDICTOR", 8, 8),
        ]
    )
    specs: list[ConnectionSpec] = []
    for source, target, fan_out, weight in (
        ("PROMPT", "HIDDEN", 16, DEFAULT_WEIGHT),
        ("REWARD_WORD", "HIDDEN", 16, DEFAULT_WEIGHT),
        ("HIDDEN", "HIDDEN", 8, RECURRENT_SEED_WEIGHT),
    ):
        specs.append(ConnectionSpec(source, target, FanOutSpec(fan_out), weight))
    # Fan-in, not fan-out - a small target (unlike HIDDEN above) would
    # otherwise saturate toward full density - see FanOutSpec.
    specs.append(ConnectionSpec("HIDDEN", "OUTPUT", FanInSpec(24), OUTPUT_SEED_WEIGHT))
    specs.append(ConnectionSpec("HIDDEN", "REWARD", FanInSpec(4), REWARD_SEED_WEIGHT))
    specs.append(ConnectionSpec("HIDDEN", "PREDICTOR", FanInSpec(4), PREDICTOR_SEED_WEIGHT))

    rng = np.random.default_rng(seed)

    synapses = build_synapses(layout, specs, rng)
    synapses.renormalize(layout, layout.population("PREDICTOR"), predictor_renormalize_total)
    snn = SNN.build(layout, synapses)
    snn.neurons.refractory_ticks = refractory_ticks

    prompt_channel = FeatureInChannel(
        layout.population("PROMPT"),
        PopulationEncoder(rng, amplitude),
        duration=prompt_duration,
    )
    word_channel = FeatureInChannel(layout.population("REWARD_WORD"), PopulationEncoder(rng, amplitude))
    output_channel = FeatureOutChannel(layout.population("OUTPUT"), PopulationDecoder(3.5))
    reward_channel = NumericChannel(layout.population("REWARD"), rng)
    predictor_channel = NumericChannel(layout.population("PREDICTOR"), rng)
    rpe = RPE(reward_channel, predictor_channel)
    stdp = Hebbian(snn, third_factor=rpe, learning_rate=learning_rate)
    homeostatic = InhibitoryPlasticity(snn, target_rate=target_rate)
    background = BackgroundDrive([layout.population("HIDDEN")], rng)

    session = Session(
        snn,
        pre=[prompt_channel, word_channel, reward_channel, background],
        post=[output_channel, reward_channel, predictor_channel, rpe, stdp, homeostatic],
    )

    setup = TurnTakingSetup(
        session,
        prompt_channel,
        word_channel,
        output_channel,
        reward_channel,
        predictor_channel,
        stdp,
        rng,
        cancel_on_interrupt,
        TrialRecorder(),
        specs=specs,
        prompt_duration=prompt_duration,
    )
    attach_predictor_diagnostic(setup)
    return setup


def _print_report(seed: int, label: str, results: list[tuple[EpochStats, PredictorReport]]):
    print(f"\n[seed={seed} {label}]")
    print(
        f"{'epoch':>5} {'turns':>6} {'resp%':>6} {'latency':>8} {'given':>6} {'cancelled':>10} "
        f"{'w(H->OUT)':>10} {'w(H->H)':>9} {'p_std':>8} {'bias':>8} {'rpe_red':>8} {'disc_p':>8} {'disc_r':>8}"
    )
    for epoch, predictor in results:
        latency = f"{epoch.mean_response_latency:.2f}" if epoch.mean_response_latency is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_OUTPUT", (0, 0, 0))[1]
        h_h = epoch.weight_stats.get("HIDDEN_to_HIDDEN", (0, 0, 0))[1]
        p_std = f"{predictor.predictor_std:.4f}" if predictor.predictor_std is not None else "-"
        bias = f"{predictor.rpe_bias:.4f}" if predictor.rpe_bias is not None else "-"
        rpe_red = f"{predictor.rpe_reduction:.4f}" if predictor.rpe_reduction is not None else "-"
        disc_p = f"{predictor.discrimination_prompt:.4f}" if predictor.discrimination_prompt is not None else "-"
        disc_r = f"{predictor.discrimination_response:.4f}" if predictor.discrimination_response is not None else "-"
        print(
            f"{epoch.index:>5} {epoch.trials:>6} {epoch.response_rate * 100:>5.1f}% {latency:>8} "
            f"{epoch.rewarded:>6} {epoch.cancelled:>10} {h_out:>10.4f} {h_h:>9.4f} {p_std:>8} {bias:>8} {rpe_red:>8} {disc_p:>8} {disc_r:>8}"
        )
        for warning in predictor.warnings:
            print(f"      ! predictor: {warning}")


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        for cancel_on_interrupt in (True, False):
            label = "cancel-on-interrupt" if cancel_on_interrupt else "ignore-interrupt"
            setup = build_turn_taking_experiment(seed, cancel_on_interrupt=cancel_on_interrupt)
            _print_report(seed, label, run_epochs(setup, 500, 40))
