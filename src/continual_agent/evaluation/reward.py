"""Shared reward, baseline, and third-factor accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from .event_stream import AccountedEvent


@dataclass
class RewardLedger:
    """Apply each reward once and convert it to a baseline-centered RPE."""

    baseline: float = 0.0
    baseline_rate: float = 0.05
    records: list[AccountedEvent] = field(default_factory=list)
    _settled: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.baseline_rate <= 1.0:
            raise ValueError("baseline_rate must be in [0, 1]")

    def settled(self, record: AccountedEvent) -> bool:
        return id(record) in self._settled

    def commit(self, reward: float, *, record: AccountedEvent | None = None) -> float:
        if record is not None:
            identity = id(record)
            if identity in self._settled:
                return 0.0
            self._settled.add(identity)
            self.records.append(record)
        rpe = reward - self.baseline
        self.baseline += self.baseline_rate * rpe
        return rpe

    def settle(self, records: list[AccountedEvent]) -> float:
        pending = [record for record in records if id(record) not in self._settled]
        if not pending:
            return 0.0
        total = sum(record.reward for record in pending)
        rpe = total - self.baseline
        self.baseline += self.baseline_rate * rpe
        for record in pending:
            identity = id(record)
            self._settled.add(identity)
            self.records.append(record)
        return rpe
