"""Shared PREDICTOR diagnostic, usable by any experiment built on RPE/Hebbian
(turn_taking.py, discrimination.py, ...). A per-tick Observer: it records
predictor.value for the trial's whole duration as one plain array, plus the
network's RPE at every outcome cue (REWARD/PUNISH/CANCELLED - a trial can
hit more than one, e.g. repeated punishment) keyed by the tick it happened
at, and hands both back as a single typed PredictorExtras under
extras["predictor"]. predictor[i] is the value at relative tick i+1 (there's
no entry for tick 0 - no tick has run yet at that instant).

report() turns one epoch's worth of Trials into a small PredictorReport - a
few headline measures plus any warnings a quick health check raised, each
aimed at a distinct failure mode:
  - predictor_std: is the signal alive at all (a literal ~0 means dead).
  - rpe_reduction: does the predictor do any better than the simplest
    possible baseline - a constant equal to the epoch's mean realized
    reward. This is what actually catches "parked at a plausible-looking
    constant" - a flat predictor scores no better than the null baseline
    regardless of which constant it happens to sit at, unlike a raw RPE
    magnitude, which a lucky constant can fake.
  - rpe_bias: the *signed* mean outcome RPE - a persistent over/under-
    prediction, distinct from noisy-but-unbiased (which rpe_reduction alone
    can't tell apart from a systematic offset).
  - discrimination_prompt / discrimination_response: does predictor value
    actually differ between trials that turn out rewarded vs. punished/
    cancelled, measured at two points in time. Comparing the two catches
    "hasn't learned this content at all" (neither differs) vs. "learned it,
    but only right before the outcome, not earlier" (response differs,
    prompt doesn't) as distinct signatures, rather than one number that
    can't tell them apart.
Cross-epoch failure modes (a ceiling that's plateaued too early, or a
measure that's unstable epoch to epoch) need history this single-epoch
report doesn't have, so they're left to whatever later aggregates multiple
epochs' reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...interaction.channels import NumericChannel
from ...interaction.plasticity import RewardSignal
from ...interaction.plugins import Observer
from ...network.snn import Spikes
from .trial import Cue, Diagnostic, Trial

OUTCOME_CUES = (Cue.REWARD, Cue.PUNISH, Cue.CANCELLED)

# Heuristic thresholds for the health checks below - deliberately coarse,
# easy to retune once we have real runs to calibrate against.
FROZEN_STD_THRESHOLD = 1e-4
SATURATION_VALUE = 0.9
SATURATION_FRACTION = 0.2
DISCRIMINATION_NOISE_RATIO = 0.1
MIN_GROUP_TRIALS = 3


@dataclass
class PredictorExtras:
    predictor: list[float]
    outcome_rpe: dict[int, float]


@dataclass
class PredictorDiagnostic(Observer, Diagnostic):
    predictor_channel: NumericChannel
    rpe: RewardSignal
    # RewardSignal only guarantees .value (see the Protocol) - discount is
    # RPE-specific, so it's passed in separately rather than assumed here.
    discount: float = 0.95

    _values: list[float] = field(init=False, default_factory=list[float])
    _outcome_rpe: dict[int, float] = field(init=False, default_factory=dict[int, float])

    def observe(self, _: Spikes):
        self._values.append(self.predictor_channel.value)

    def start(self):
        self._values = []
        self._outcome_rpe = {}

    def mark(self, cue: Cue, tick: int):
        if cue in OUTCOME_CUES:
            self._outcome_rpe[tick] = self.rpe.value

    def finalize(self) -> dict[str, Any]:
        return {"predictor": PredictorExtras(self._values, self._outcome_rpe)}


def attach(setup: Any) -> PredictorDiagnostic:
    """Build a PredictorDiagnostic for `setup` (any experiment setup with the
    usual .session/.predictor_channel/.stdp/.trial_recorder fields), wire it
    into the session (for per-tick sampling) and the trial recorder's
    diagnostics (for start/mark/finalize)."""
    discount = getattr(setup.stdp.third_factor, "discount", 0.95)
    diagnostic = PredictorDiagnostic(setup.predictor_channel, setup.stdp.third_factor, discount)
    setup.session.post.append(diagnostic)
    setup.trial_recorder.diagnostics.append(diagnostic)
    return diagnostic


def value_at(trial: Trial, cue: Cue) -> float | None:
    """predictor.value at the (first) tick `cue` was marked, or None if the
    cue didn't occur, the diagnostic wasn't attached, or it occurred before
    any tick had run."""
    extras = trial.extras.get("predictor")
    tick = trial.cue_tick(cue)
    if not isinstance(extras, PredictorExtras) or tick is None or tick < 1 or tick > len(extras.predictor):
        return None
    return extras.predictor[tick - 1]


def _realized_value(trial: Trial) -> float | None:
    return trial.extras.get("reward_value", trial.extras.get("punish_value"))


@dataclass
class PredictorReport:
    trials: int
    predictor_std: float | None
    rpe_bias: float | None
    rpe_reduction: float | None
    discrimination_prompt: float | None
    discrimination_response: float | None
    warnings: list[str] = field(default_factory=list[str])


def report(trials: list[Trial], discount: float = 0.95) -> PredictorReport:
    """Summarize one epoch's worth of Trials into headline PREDICTOR measures
    plus any health-check warnings."""
    predictor_values: list[float] = []
    # (actual outcome RPE, realized reward/punish magnitude at that trial)
    rpe_pairs: list[tuple[float, float]] = []
    for t in trials:
        extras = t.extras.get("predictor")
        if not isinstance(extras, PredictorExtras):
            continue
        predictor_values.extend(extras.predictor)
        realized = _realized_value(t)
        if realized is not None:
            rpe_pairs.extend((rpe, realized) for rpe in extras.outcome_rpe.values())

    predictor_std = float(np.std(predictor_values)) if predictor_values else None

    rpe_bias = None
    rpe_reduction = None
    if rpe_pairs:
        actual_rpe, realized = (np.array(v) for v in zip(*rpe_pairs))
        rpe_bias = float(np.mean(actual_rpe))
        # The simplest possible baseline: predict the epoch's mean realized
        # reward, always. RPE_null follows the same TD formula with a
        # constant prediction c: reward + discount*c - c = reward - c*(1-discount).
        baseline = float(np.mean(realized))
        null_rpe = realized - baseline * (1.0 - discount)
        null_abs = float(np.mean(np.abs(null_rpe)))
        if null_abs > 1e-9:
            actual_abs = float(np.mean(np.abs(actual_rpe)))
            rpe_reduction = 1.0 - actual_abs / null_abs

    def group_values(cue: Cue, outcome: Cue | tuple[Cue, ...]) -> list[float]:
        outcomes = outcome if isinstance(outcome, tuple) else (outcome,)
        return [v for t in trials if any(t.has_cue(o) for o in outcomes) for v in [value_at(t, cue)] if v is not None]

    rewarded_count = sum(1 for t in trials if t.has_cue(Cue.REWARD))
    punished_count = sum(1 for t in trials if t.has_cue(Cue.PUNISH) or t.has_cue(Cue.CANCELLED))

    def discrimination_at(cue: Cue) -> float | None:
        rewarded = group_values(cue, Cue.REWARD)
        punished = group_values(cue, (Cue.PUNISH, Cue.CANCELLED))
        if not rewarded or not punished:
            return None
        return float(np.mean(rewarded) - np.mean(punished))

    discrimination_prompt = discrimination_at(Cue.PROMPT)
    discrimination_response = discrimination_at(Cue.RESPONSE)

    warnings: list[str] = []
    if predictor_std is not None and predictor_std < FROZEN_STD_THRESHOLD:
        warnings.append(f"predictor variance is ~0 ({predictor_std:.2e}) - signal may be frozen/disconnected")
    if predictor_values:
        saturated = sum(1 for v in predictor_values if abs(v) >= SATURATION_VALUE) / len(predictor_values)
        if saturated >= SATURATION_FRACTION:
            warnings.append(f"predictor saturating near its value range ({saturated:.0%} of ticks |value|>={SATURATION_VALUE})")
    if rpe_reduction is not None and rpe_reduction <= 0:
        warnings.append(f"predictor is no better than a constant baseline at predicting reward (rpe_reduction={rpe_reduction:.2f})")

    if 0 < rewarded_count < MIN_GROUP_TRIALS or 0 < punished_count < MIN_GROUP_TRIALS:
        warnings.append(f"too few rewarded ({rewarded_count}) / punished ({punished_count}) trials to assess discrimination reliably")
    elif rewarded_count >= MIN_GROUP_TRIALS and punished_count >= MIN_GROUP_TRIALS and predictor_std:
        noise_floor = DISCRIMINATION_NOISE_RATIO * predictor_std
        if discrimination_response is not None and discrimination_response < -noise_floor:
            warnings.append("predictor discrimination is inverted - predicts higher value for eventually-punished trials")
        elif discrimination_response is not None and abs(discrimination_response) < noise_floor:
            warnings.append("predictor shows no measurable discrimination between eventually-rewarded and eventually-punished trials")
        elif discrimination_prompt is not None and abs(discrimination_prompt) < noise_floor:
            warnings.append("predictor only differentiates outcome close to the response, not earlier at prompt")

    return PredictorReport(
        len(trials),
        predictor_std,
        rpe_bias,
        rpe_reduction,
        discrimination_prompt,
        discrimination_response,
        warnings,
    )
