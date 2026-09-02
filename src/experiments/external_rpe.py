"""A ground-truth stand-in for interaction.stdp.RPE, used with intent-gating.

We found that a plasticity-magnitude deadzone can't reliably separate genuine
reward signal from ambient network noise: reading the reward-channel value
back from spikes means RPE can't tell "this changed because I deliberately
forced it" from "this changed because of incidental correlated activity" -
the two distributions overlap too much across seeds to threshold apart.

Rather than infer intent from a noisy readout, this reports the exact value
we intend to deliver directly - we already know it with certainty at the
moment we call force(). It implements the same `calculate_rpe() -> float`
protocol as RPE, so it's a drop-in third_factor for STDP; the neural REWARD
population and NumericChannel are left fully wired up and still driven by
force() (so the network's structure and dynamics are unchanged), but their
resulting spikes are not read for learning - only for whatever ambient
activity the experiment chooses to observe/report separately.

Note this does NOT diff against the previously notified value the way RPE
diffs against a fluctuating reading. Diffing makes sense when detecting
surprise in a naturally-varying signal, but every notify() here is a fully
known, intentional outcome (we always deliver exactly reward_spike or exactly
0.0) - diffing a repeated constant against itself is zero after the first
occurrence, which silently extinguishes all learning after one event. The
notified value *is* the reinforcement signal, every time.

This is temporary scaffolding to verify that intent-gated learning works at
all before deciding whether (and how) to teach the network to read a genuine
value out of its own reward population.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExternalRPE:
    last_rpe: float = field(init=False, default=0.0)

    _pending_value: float | None = field(init=False, default=None, repr=False)

    def notify(self, value: float):
        """Record the value the next calculate_rpe() call should report."""
        self._pending_value = value

    def calculate_rpe(self) -> float:
        value = self._pending_value if self._pending_value is not None else 0.0
        self.last_rpe = value
        self._pending_value = None
        return value
