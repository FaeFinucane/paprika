def ramp(step: float, length: float, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over `length` steps, held at `end`
    after - used to make behaviour tests stricter as training progresses
    instead of applying full strictness (and punishment) from step one."""
    if length <= 0:
        return end
    t = min(1.0, step / length)
    return start + (end - start) * t


def evidence_ramp(score: float, target: float, start: float, end: float) -> float:
    """Like ramp(), but `score` is a measured competence score that can be
    negative (no evidence yet, or a net-bad track record) - clamped to 0
    first, so "no evidence" reads as fully lenient, not past it."""
    return ramp(max(0.0, score), target, start, end)


def duration_reward(duration: int, ideal: int, base: float, tolerance: float) -> float:
    """Peaks at ideal duration, goes negative beyond `tolerance` ticks off -
    not floored at 0, so overshoot actively depresses rather than just
    under-reinforces."""
    return base * (1.0 - abs(duration - ideal) / tolerance)
