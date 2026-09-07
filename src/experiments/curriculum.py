def ramp(step: int, length: int, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over `length` steps, held at `end`
    after - used to make behaviour tests stricter as training progresses
    instead of applying full strictness (and punishment) from step one."""
    if length <= 0:
        return end
    t = min(1.0, step / length)
    return start + (end - start) * t


def duration_reward(duration: int, ideal: int, base: float, tolerance: float) -> float:
    """Peaks at ideal duration, goes negative beyond `tolerance` ticks off -
    not floored at 0, so overshoot actively depresses rather than just
    under-reinforces."""
    return base * (1.0 - abs(duration - ideal) / tolerance)
