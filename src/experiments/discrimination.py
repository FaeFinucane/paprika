"""Discrimination learning: N distinct cues, each requiring a different
learned response. Tests whether reward-modulated STDP can hold several
associations in the same shared HIDDEN population without them overwriting
each other - the natural next milestone after single-pattern call/response,
memory (TD prediction), and punishment (inhibitory plasticity).
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
from .curriculum import duration_reward, evidence_ramp
from .diagnostics import Cue, EpochStats, PredictorReport, Trial, TrialRecorder, attach_predictor_diagnostic, predictor_report, summarize_epoch

REWARD_WORDS: tuple[str, ...] = ("yes", "good")
NO_WORD: str = "no"
CUES: tuple[str, ...] = tuple(f"CUE_{letter}" for letter in "ABCDEF")
TOKENS: tuple[str, ...] = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX")
CUE_TOKEN: dict[str, str] = dict(zip(CUES, TOKENS))


@dataclass
class DiscriminationSetup:
    session: Session
    speech_in: FeatureInChannel
    speech_out: FeatureOutChannel
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
    tolerance_start: float = 4.0
    tolerance_end: float = 1.0
    max_duration: int = 15
    cancel_penalty: float = -0.75
    incorrect_penalty: float = -0.75
    content_window: int = 40
    no_response_score: float = -0.5
    ramp_target_competence: float = 0.5
    max_punishment_repeats: int = 3
    wait_min: int = 5
    wait_max: int = 20
    cue_duration: int = 3
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


def recent_response_rate(setup: DiscriminationSetup) -> float:
    recent = setup.trial_recorder.trials[-setup.content_window:]
    if not recent:
        return 0.0
    return sum(t.has_cue(Cue.RESPONSE) for t in recent) / len(recent)


def recent_duration_score(setup: DiscriminationSetup) -> float:
    recent = setup.trial_recorder.trials[-setup.content_window:]
    if not recent:
        return setup.no_response_score

    def score(t: Trial) -> float:
        duration = t.extras.get("response_duration")
        if duration is None:
            return setup.no_response_score
        # ideal_duration, not max_duration - normalizing by max_duration=15
        # made a duration=1 response (as far as possible from ideal=3) score
        # 0.87, reading as "great" when it's actually the worst outcome -
        # this is what let tolerance tighten to its strictest value despite
        # duration never actually being close to ideal (see conversation).
        return 1.0 - min(1.0, abs(duration - setup.ideal_duration) / setup.ideal_duration)

    return sum(score(t) for t in recent) / len(recent)


def start_trial(setup: DiscriminationSetup):
    setup.trial_recorder.start(setup.session.snn.tick)


def mark(setup: DiscriminationSetup, cue: Cue):
    setup.trial_recorder.mark(cue, setup.session.snn.tick)


def finalize_trial(setup: DiscriminationSetup) -> Trial:
    return setup.trial_recorder.finalize(setup.session.snn.tick)


def step(setup: DiscriminationSetup):
    spikes = setup.session.tick()
    setup.stdp.update(setup.session.snn)
    return spikes


def deliver_no(setup: DiscriminationSetup, penalty: float) -> bool:
    """Say 'no' letter by letter, delivering competence-scaled punishment on
    the last letter."""
    for character in NO_WORD:
        setup.speech_in.write(character)

    remaining = len(NO_WORD)
    interrupted = False
    while remaining > 0:
        step(setup)
        active = setup.speech_out.current is not None
        if active and setup.speech_out.has_feature_changed:
            interrupted = True
        remaining -= 1

    value = penalty * evidence_ramp(recent_response_rate(setup), setup.ramp_target_competence, 0.0, 1.0)
    setup.reward_channel.force(value, duration=3)
    step(setup)
    mark(setup, Cue.PUNISH)
    setup.trial_recorder.set_extra("punish_value", value)
    return interrupted


def punish(setup: DiscriminationSetup, penalty: float):
    interrupted = True
    repeats = 0
    while interrupted and repeats < setup.max_punishment_repeats:
        interrupted = deliver_no(setup, penalty)
        repeats += 1


def deliver_word(setup: DiscriminationSetup, word: str) -> tuple[str, bool]:
    """Feed `word` letter by letter. If interrupted and cancel_on_interrupt
    is set, switch to punish() instead - itself repeated if interrupted too.
    Returns the word actually delivered and whether it was cancelled."""
    for character in word:
        setup.speech_in.write(character)

    remaining = len(word)
    cancelled = False
    while remaining > 0:
        step(setup)
        active = setup.speech_out.current is not None
        just_emitted = active and setup.speech_out.has_feature_changed
        remaining -= 1
        if just_emitted and setup.cancel_on_interrupt and not cancelled:
            setup.speech_in.clear()
            cancelled = True

    if cancelled:
        punish(setup, setup.cancel_penalty)
        return NO_WORD, True
    return word, False


def run_trial(setup: DiscriminationSetup) -> Trial:
    start_trial(setup)

    if setup.wait_max > 0:
        wait_ticks = int(setup.rng.integers(setup.wait_min, setup.wait_max + 1))
        for _ in range(wait_ticks):
            step(setup)
            active = setup.speech_out.current is not None
            if active and setup.speech_out.has_feature_changed:
                # punish(setup, setup.cancel_penalty)
                break
    mark(setup, Cue.BASELINE)

    cue = str(setup.rng.choice(CUES))
    setup.speech_in.write(cue)
    setup.trial_recorder.set_extra("cue", cue)

    response_token = None
    for i in range(setup.cue_duration + setup.response_timeout):
        step(setup)
        if i == 0:
            # After the cue has had one tick to actually reach HIDDEN and
            # propagate back, not at the instant of writing it - see
            # turn_taking.py's identical fix.
            mark(setup, Cue.PROMPT)
        current = setup.speech_out.current
        if current is not None and setup.speech_out.has_feature_changed:
            response_token = current.feature
            break
    setup.speech_in.clear()

    if response_token is not None:
        mark(setup, Cue.RESPONSE)
        setup.trial_recorder.set_extra("response_token", response_token)
        setup.trial_recorder.set_extra("correct", response_token == CUE_TOKEN[cue])

        # Reward/punishment isn't delivered until output actually stops - a
        # real consequence of the delay is that "stop outputting" becomes a
        # precondition for any consequence at all, not something rewarded
        # directly.
        duration = 1
        for _ in range(setup.max_duration - 1):
            step(setup)
            if setup.speech_out.current is None:
                break
            duration += 1
        setup.trial_recorder.set_extra("response_duration", duration)

        tolerance = evidence_ramp(recent_duration_score(setup), setup.ramp_target_competence, setup.tolerance_start, setup.tolerance_end)
        shaped_reward = duration_reward(duration, setup.ideal_duration, setup.reward_spike, tolerance)

        if response_token == CUE_TOKEN[cue]:
            value = shaped_reward
        else:
            content_weight = evidence_ramp(recent_response_rate(setup), setup.ramp_target_competence, 0.0, 1.0)
            value = shaped_reward * (1.0 - content_weight) + setup.incorrect_penalty * content_weight

        if value >= 0:
            word = str(setup.rng.choice(setup.reward_words))
            delivered_word, cancelled = deliver_word(setup, word)
            if not cancelled:
                setup.reward_channel.force(value, duration=3)
                step(setup)
                mark(setup, Cue.REWARD)
                setup.trial_recorder.set_extra("reward_value", value)
                setup.trial_recorder.set_extra("delivered_word", delivered_word)
            else:
                mark(setup, Cue.CANCELLED)
        else:
            punish(setup, value)

    for _ in range(setup.cooldown):
        step(setup)

    return finalize_trial(setup)


def run(setup: DiscriminationSetup, ticks: int):
    end_tick = setup.session.snn.tick + ticks
    while setup.session.snn.tick < end_tick:
        run_trial(setup)
    return setup


def run_epoch(setup: DiscriminationSetup, ticks: int, index: int = 0) -> tuple[EpochStats, PredictorReport]:
    start_tick = setup.session.snn.tick
    start_trial_count = len(setup.trial_recorder.trials)

    run(setup, ticks)

    epoch_trials = setup.trial_recorder.trials[start_trial_count:]
    stats = summarize_epoch(epoch_trials, index, start_tick, setup.session.snn.tick, setup.weight_stats())
    discount = getattr(setup.stdp.third_factor, "discount", 0.95)
    return stats, predictor_report(epoch_trials, discount)


def run_epochs(setup: DiscriminationSetup, ticks_per_epoch: int, epochs: int) -> list[tuple[EpochStats, PredictorReport]]:
    return [run_epoch(setup, ticks_per_epoch, index) for index in range(epochs)]


def build_discrimination_experiment(
    seed: int = 0,
    *,
    cancel_on_interrupt: bool = False,
    amplitude: float = 1.0,
    cue_duration: int = 3,
    learning_rate: float = 0.001,
    target_rate: float = 0.005,
    # Measured empirically, not derived - 1.25 (naive threshold*1.25) leaves
    # the network barely able to respond at all (~3-5% response rate); 4.0
    # gives healthy, stable responsiveness (~85-95%) with no further benefit
    # from going higher (see conversation).
    renormalize_total: float = 4.0,
    # SPEECH_OUT was structurally biased toward a subset of tokens (some
    # completely unreachable) before this; PREDICTOR was too weakly driven by
    # HIDDEN to develop any real reward-anticipation signal (mostly noise) -
    # both measured empirically, see conversation.
    speech_out_renormalize_total: float = 1.5,
    predictor_renormalize_total: float = 4.0,
) -> DiscriminationSetup:
    letters = tuple(sorted(set("".join(REWARD_WORDS) + NO_WORD)))
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("SPEECH_IN", (*CUES, *letters), 4),
            FeaturePopulationSpec("SPEECH_OUT", TOKENS, 6),
            NeuronPopulationSpec("HIDDEN", 160, inhibitory=0.2),
            NumericPopulationSpec("REWARD", 8, 8),
            NumericPopulationSpec("PREDICTOR", 8, 8),
        ]
    )
    specs = [
        ConnectionSpec("SPEECH_IN", "HIDDEN", FanOutSpec(8), WeightSpec(0.6, 0.04)),
        ConnectionSpec("HIDDEN", "HIDDEN", FanOutSpec(8), WeightSpec(0.3, 0.05)),
        ConnectionSpec("HIDDEN", "SPEECH_OUT", FanInSpec(8), WeightSpec(0.7, 0.2)),
        ConnectionSpec("HIDDEN", "REWARD", FanInSpec(8), WeightSpec(0.08, 0.01)),
        ConnectionSpec("HIDDEN", "PREDICTOR", FanInSpec(8), WeightSpec(0.08, 0.01)),
    ]

    rng = np.random.default_rng(seed)

    synapses = build_synapses(layout, specs, rng)
    synapses.renormalize(layout, layout.population("HIDDEN"), renormalize_total)
    synapses.renormalize(layout, layout.population("SPEECH_OUT"), speech_out_renormalize_total)
    synapses.renormalize(layout, layout.population("PREDICTOR"), predictor_renormalize_total)
    # REWARD is deliberately left un-renormalized - it needs to stay weakly
    # HIDDEN-driven so its forced ground-truth value stays authoritative,
    # unlike PREDICTOR which is never forced and needs real HIDDEN-driven
    # signal to learn anything at all (see conversation).
    snn = SNN.build(layout, synapses)

    speech_in = FeatureInChannel(
        layout.population("SPEECH_IN"),
        PopulationEncoder(rng, amplitude),
        duration=cue_duration,
    )
    speech_out = FeatureOutChannel(layout.population("SPEECH_OUT"), PopulationDecoder(3.5))
    reward_channel = NumericChannel(layout.population("REWARD"), rng)
    predictor_channel = NumericChannel(layout.population("PREDICTOR"), rng)
    rpe = RPE(reward_channel, predictor_channel)
    stdp = Hebbian(snn, third_factor=rpe, learning_rate=learning_rate)
    homeostatic = InhibitoryPlasticity(snn, target_rate=target_rate)

    background = BackgroundDrive([layout.population("HIDDEN")], rng, amplitude=0.4)

    session = Session(
        snn,
        # No predictor_background - now that PREDICTOR is renormalized, HIDDEN
        # alone reliably drives it, and background noise was polluting RPE's
        # tick-to-tick predictor.value difference more than it helped (see
        # conversation).
        pre=[speech_in, reward_channel, background],
        post=[speech_out, reward_channel, predictor_channel, rpe, stdp, homeostatic],
    )

    setup = DiscriminationSetup(
        session,
        speech_in,
        speech_out,
        reward_channel,
        predictor_channel,
        stdp,
        rng,
        cancel_on_interrupt,
        TrialRecorder(),
        specs=specs,
        cue_duration=cue_duration,
    )
    attach_predictor_diagnostic(setup)
    return setup


def _print_report(seed: int, results: list[tuple[EpochStats, PredictorReport]]):
    print(f"\n[seed={seed}]")
    print(
        f"{'epoch':>5} {'trials':>6} {'resp%':>6} {'latency':>8} {'given':>6} {'cancelled':>10} "
        f"{'w(H->OUT)':>10} {'p_std':>8} {'bias':>8} {'rpe_red':>8} {'disc_p':>8} {'disc_r':>8}"
    )
    for epoch, predictor in results:
        latency = f"{epoch.mean_response_latency:.2f}" if epoch.mean_response_latency is not None else "-"
        h_out = epoch.weight_stats.get("HIDDEN_to_SPEECH_OUT", (0, 0, 0))[1]
        p_std = f"{predictor.predictor_std:.4f}" if predictor.predictor_std is not None else "-"
        bias = f"{predictor.rpe_bias:.4f}" if predictor.rpe_bias is not None else "-"
        rpe_red = f"{predictor.rpe_reduction:.4f}" if predictor.rpe_reduction is not None else "-"
        disc_p = f"{predictor.discrimination_prompt:.4f}" if predictor.discrimination_prompt is not None else "-"
        disc_r = f"{predictor.discrimination_response:.4f}" if predictor.discrimination_response is not None else "-"
        print(
            f"{epoch.index:>5} {epoch.trials:>6} {epoch.response_rate * 100:>5.1f}% {latency:>8} "
            f"{epoch.rewarded:>6} {epoch.cancelled:>10} {h_out:>10.4f} {p_std:>8} {bias:>8} {rpe_red:>8} {disc_p:>8} {disc_r:>8}"
        )
        for warning in predictor.warnings:
            print(f"      ! predictor: {warning}")


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(runs):
        seed = random.randrange(2**32)
        setup = build_discrimination_experiment(seed)
        _print_report(seed, run_epochs(setup, 500, 40))
