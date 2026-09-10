"""Dopamine-modulated three-factor STDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ...network.adjustments import NetworkAdjustment
from ...network.snn import SNN, Spikes
from ..plugins import SpikeAdaptation


class ModulatorySignal(Protocol):
    @property
    def value(self) -> float: ...


@dataclass
class DopamineSTDP(SpikeAdaptation):
    snn: SNN
    dopamine: ModulatorySignal
    learning_rate: float = 0.01
    trace_decay: float = 0.95
    deadzone: float = 0.0
    maximum_modulation: float = 1.0
    eligibility: np.ndarray = field(init=False)
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)
    plastic: np.ndarray = field(init=False)
    response_sign: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.learning_rate < 0
            or not 0 <= self.trace_decay <= 1
            or self.deadzone < 0
            or self.maximum_modulation <= 0
        ):
            raise ValueError("invalid dopamine STDP configuration")
        count = self.snn.synapses.source.size
        self.eligibility = np.zeros(count)
        self.pre_trace = np.zeros(count)
        self.post_trace = np.zeros(count)
        self.plastic = self.snn.synapses.policy_mask("dopamine_stdp")
        self.response_sign = self.snn.layout.dopamine_response_sign[self.snn.synapses.target]

    def observe(self, spikes: Spikes) -> None:
        pre = spikes.values[self.snn.synapses.source]
        post = spikes.values[self.snn.synapses.target]
        self.eligibility *= self.trace_decay
        self.eligibility += self.pre_trace * post - self.post_trace * pre
        self.pre_trace = self.trace_decay * self.pre_trace + pre
        self.post_trace = self.trace_decay * self.post_trace + post

    def propose(self, snn: SNN) -> NetworkAdjustment:
        if snn is not self.snn:
            raise ValueError("dopamine STDP belongs to another network")
        modulation = float(self.dopamine.value)
        if not np.isfinite(modulation):
            raise ValueError("dopamine modulation must be finite")
        magnitude = abs(modulation)
        if magnitude <= self.deadzone:
            return NetworkAdjustment()
        modulation = np.sign(modulation) * min(
            self.maximum_modulation,
            (magnitude - self.deadzone) / max(1.0 - self.deadzone, np.finfo(float).eps),
        )
        mask = self.plastic & (self.response_sign != 0)
        return NetworkAdjustment(
            self.learning_rate * modulation * self.eligibility * self.response_sign,
            mask,
        )

    def reset(self) -> None:
        self.eligibility.fill(0.0)
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
