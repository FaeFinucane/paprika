"""Generic epoch-level reporting derived from Trial records, shared across
experiments so response rate, reward/punishment counts, and latency are all
computed the same way everywhere. Anything experiment-specific (like
discrimination's response-token accuracy) stays out of here and is read
straight off Trial.extras by that experiment's own report code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trial import Cue, Trial


@dataclass
class EpochStats:
    index: int
    start_tick: int
    end_tick: int
    trials: int
    responded: int
    rewarded: int
    punished: int
    cancelled: int
    mean_response_latency: float | None
    weight_stats: dict[str, tuple[float, float, float]]

    @property
    def response_rate(self) -> float:
        return self.responded / self.trials if self.trials else 0.0


def summarize_epoch(
    trials: list[Trial],
    index: int,
    start_tick: int,
    end_tick: int,
    weight_stats: dict[str, tuple[float, float, float]],
) -> EpochStats:
    responded = [t for t in trials if t.has_cue(Cue.RESPONSE)]
    # -1: the earliest a response can appear is prompt_tick + 1 (SNN.step()
    # increments tick before returning), so 0 means "as early as possible".
    latencies: list[int] = []
    for t in responded:
        prompt_tick, response_tick = t.cue_tick(Cue.PROMPT), t.cue_tick(Cue.RESPONSE)
        if prompt_tick is not None and response_tick is not None:
            latencies.append(response_tick - prompt_tick - 1)
    return EpochStats(
        index,
        start_tick,
        end_tick,
        len(trials),
        len(responded),
        sum(t.count_cue(Cue.REWARD) for t in trials),
        sum(t.count_cue(Cue.PUNISH) for t in trials),
        sum(t.count_cue(Cue.CANCELLED) for t in trials),
        (sum(latencies) / len(latencies)) if latencies else None,
        weight_stats,
    )
