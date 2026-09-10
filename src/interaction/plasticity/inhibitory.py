"""Reward-independent inhibitory homeostatic plasticity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...network.adjustments import NetworkAdjustment
from ...network.snn import SNN, Spikes
from ..plugins import SpikeAdaptation


@dataclass
class InhibitoryHomeostasis(SpikeAdaptation):
    snn: SNN
    target_rate: float
    learning_rate: float = 0.001
    trace_decay: float = 0.95
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)
    plastic: np.ndarray = field(init=False)
    depression: float = field(init=False)
    _pending_delta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if (
            not 0 <= self.target_rate <= 1
            or self.learning_rate < 0
            or not 0 <= self.trace_decay < 1
        ):
            raise ValueError("invalid inhibitory-homeostasis configuration")
        count = self.snn.synapses.source.size
        self.pre_trace = np.zeros(count)
        self.post_trace = np.zeros(count)
        self.plastic = self.snn.synapses.policy_mask("inhibitory_homeostatic")
        self.depression = 2.0 * self.target_rate / (1.0 - self.trace_decay)
        self._pending_delta = np.zeros(count)

    def observe(self, spikes: Spikes) -> None:
        pre = spikes.values[self.snn.synapses.source]
        post = spikes.values[self.snn.synapses.target]
        self._pending_delta.fill(0.0)
        self._pending_delta[self.plastic & pre] += (
            self.post_trace[self.plastic & pre] - self.depression
        )
        self._pending_delta[self.plastic & post] += self.pre_trace[self.plastic & post]
        self.pre_trace = self.trace_decay * self.pre_trace + pre
        self.post_trace = self.trace_decay * self.post_trace + post

    def propose(self, snn: SNN) -> NetworkAdjustment:
        if snn is not self.snn:
            raise ValueError("inhibitory homeostasis belongs to another network")
        # Strengthening an inhibitory projection means increasing its positive
        # magnitude. The source trait supplies the negative current sign.
        return NetworkAdjustment(self.learning_rate * self._pending_delta, self.plastic)

    def reset(self) -> None:
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
