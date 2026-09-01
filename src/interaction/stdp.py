from dataclasses import dataclass, field

import numpy as np

from ..network.snn import SNN, Spikes

@dataclass
class Learning:
    synapse_count: int
    learning_rate: float = 0.01
    trace_decay: float = 0.95
    gamma: float = 0.99
    max_rpe: float = 1.0
    eligibility: np.ndarray = field(init=False)
    pre: np.ndarray = field(init=False)
    post: np.ndarray = field(init=False)
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)
    accounted: set[str] = field(default_factory=set[str])

    def __post_init__(self):
        if (
            self.synapse_count < 0
            or self.learning_rate < 0
            or not 0 <= self.trace_decay <= 1
            or not 0 <= self.gamma <= 1
            or self.max_rpe <= 0
        ):
            raise ValueError("invalid learning configuration")
        
        self.eligibility = np.zeros(self.synapse_count)
        self.pre = np.zeros(self.synapse_count, bool)
        self.post = np.zeros(self.synapse_count, bool)
        self.pre_trace = np.zeros(self.synapse_count)
        self.post_trace = np.zeros(self.synapse_count)

    def observe(self, spikes: Spikes, source: np.ndarray, target: np.ndarray):
        self.pre[:] = spikes.values[source]
        self.post[:] = spikes.values[target]
        self.eligibility *= self.trace_decay
        self.eligibility += self.pre_trace * self.post - self.post_trace * self.pre
        self.pre_trace = self.trace_decay * self.pre_trace + self.pre
        self.post_trace = self.trace_decay * self.post_trace + self.post

    def rpe(self, reward: float, current_value: float, next_value: float, terminal: bool = False):
        return float(
            np.clip(
                reward + (0 if terminal else self.gamma * next_value) - current_value,
                -self.max_rpe,
                self.max_rpe,
            )
        )

    def update(self, snn: SNN, identity: str, reward: float, current_value: float = 0.0, next_value: float = 0.0, terminal: bool = True):
        if identity in self.accounted:
            raise ValueError("outcome already accounted")
        signal = self.rpe(reward, current_value, next_value, terminal)
        snn.apply_weight_delta(self.learning_rate * signal * self.eligibility)
        self.eligibility.fill(0)
        self.pre_trace.fill(0)
        self.post_trace.fill(0)
        self.accounted.add(identity)
        return signal
