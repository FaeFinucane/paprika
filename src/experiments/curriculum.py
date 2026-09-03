def ramp(step: int, length: int, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over `length` steps, held at `end`
    after - used to make behaviour tests stricter as training progresses
    instead of applying full strictness (and punishment) from step one."""
    if length <= 0:
        return end
    t = min(1.0, step / length)
    return start + (end - start) * t
