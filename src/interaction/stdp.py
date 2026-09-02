from dataclasses import dataclass, field
import numpy as np

from ..network.snn import SNN, Spikes
from .plugins import Observer
from .channels import NumericChannel

@dataclass
class RPE:
    reward_channel: NumericChannel

    # TODO: RPE should be in a specific range and normalized.

    last_value: float = field(init=False, default=0.0)

    def calculate_rpe(self) -> float:
        current_value = self.reward_channel.value
        rpe = current_value - self.last_value
        self.last_value = current_value
        return rpe

@dataclass
class STDP(Observer):
    """
    Implements a spike-timing-dependent plasticity (STDP) learning rule for the neural network.
    This class observes neural spikes and updates synaptic weights based on the timing of pre- and post-synaptic spikes.
    """

    snn: SNN
    third_factor: RPE | None = None

    learning_rate: float = 0.01
    trace_decay: float = 0.95

    eligibility: np.ndarray = field(init=False)
    pre: np.ndarray = field(init=False)
    post: np.ndarray = field(init=False)
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)

    def __post_init__(self):
        if (
            self.learning_rate < 0
            or not 0 <= self.trace_decay <= 1
        ):
            raise ValueError("invalid learning configuration")
        
        synapse_count = self.snn.layout.total_count
        
        self.eligibility = np.zeros(synapse_count)
        self.pre = np.zeros(synapse_count, bool)
        self.post = np.zeros(synapse_count, bool)
        self.pre_trace = np.zeros(synapse_count)
        self.post_trace = np.zeros(synapse_count)

    # TODO: Why does this method take a source and a target?
    def observe(self, spikes: Spikes):
        # Is this enough for determining elegibility?
        self.pre[:] = spikes.values[self.snn.synapses.source]
        self.post[:] = spikes.values[self.snn.synapses.target]
        self.eligibility *= self.trace_decay
        self.eligibility += self.pre_trace * self.post - self.post_trace * self.pre
        self.pre_trace = self.trace_decay * self.pre_trace + self.pre
        self.post_trace = self.trace_decay * self.post_trace + self.post

    # TODO: Weight updates should *not* just be added ad-hoc, at least not long-term.
    # TODO: Depending on how we want to make batch training work, updates should be delivered continuously every few ticks by making the STDP an Influence.
    # TODO: We'll then want a way to inject the 3rd-factor RPE in, rather than just via calling update()
    def update(self, snn: SNN):
        if self.third_factor is not None:
            rpe = self.third_factor.calculate_rpe()
        else:
            rpe = 1.0

        snn.apply_weight_delta(self.learning_rate * rpe * self.eligibility)
        self.eligibility.fill(0)
        self.pre_trace.fill(0)
        self.post_trace.fill(0)


