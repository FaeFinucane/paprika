from dataclasses import dataclass, field


@dataclass(frozen=True)
class OutputEvent:
    feature: str
    timestamp: int
    evidence: float = 0.0


@dataclass
class Turn:
    inputs: list[str] = field(default_factory=list[str])
    outputs: list[OutputEvent] = field(default_factory=list[OutputEvent])
    status: str = "open"
    error: str | None = None
    start_tick: int | None = None
    end_tick: int | None = None


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list[Turn])
    active: Turn | None = None

    def begin(self, tick: int | None = None):
        if self.active is not None:
            raise RuntimeError("turn already active")
        self.active = Turn(start_tick=tick)
        self.turns.append(self.active)
        return self.active

    def input(self, feature: str):
        if self.active is None or self.active.status != "open":
            raise RuntimeError("no open turn")
        self.active.inputs.append(feature)

    def output(self, event: OutputEvent):
        if self.active is None or self.active.status != "open":
            raise RuntimeError("no open turn")
        self.active.outputs.append(event)

    def complete(self, status: str = "eos", tick: int | None = None, error: str | None = None):
        if self.active is None or self.active.status != "open":
            raise RuntimeError("no open turn")
        if status not in {"eos", "timeout", "cancelled", "failed"}:
            raise ValueError("invalid completion status")
        self.active.status = status
        self.active.end_tick = tick
        self.active.error = error
        result = self.active
        self.active = None
        return result
