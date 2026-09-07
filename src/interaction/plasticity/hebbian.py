"""Reward-modulated (three-factor) Hebbian plasticity - excitatory synapses
only. Inhibitory synapses use a separate, reward-independent homeostatic
rule instead (see homeostatic.py) - real inhibitory plasticity isn't
dopamine-gated the way this is."""

from dataclasses import dataclass, field

import numpy as np

from ...network.snn import SNN, Spikes
from ..plugins import Observer
from .reward import RewardSignal


@dataclass
class Hebbian(Observer):
    snn: SNN
    third_factor: RewardSignal | None = None

    learning_rate: float = 0.01
    trace_decay: float = 0.95

    # Below this |rpe|, no weight update is applied at all - a plasticity
    # threshold, roughly analogous to the coincidence-detection thresholds
    # (e.g. NMDA/calcium-gated) that keep real synapses from being nudged by
    # every faint, ambiguous correlation. Without it, a third factor sampled
    # every tick applies a weight change for even negligible ambient
    # fluctuation, and since that happens far more often than genuine reward
    # events, the noise can dominate the signal purely by frequency. 0.0
    # disables the deadzone (every nonzero rpe applies an update).
    rpe_deadzone: float = 0.0

    # TODO: pre_trace/post_trace are identical to InhibitoryPlasticity's - extract and share.
    eligibility: np.ndarray = field(init=False)
    pre: np.ndarray = field(init=False)
    post: np.ndarray = field(init=False)
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)
    excitatory: np.ndarray = field(init=False)

    def __post_init__(self):
        if (
            self.learning_rate < 0
            or not 0 <= self.trace_decay <= 1
            or self.rpe_deadzone < 0
        ):
            raise ValueError("invalid learning configuration")

        synapse_count = len(self.snn.synapses.source)

        self.eligibility = np.zeros(synapse_count)
        self.pre = np.zeros(synapse_count, bool)
        self.post = np.zeros(synapse_count, bool)
        self.pre_trace = np.zeros(synapse_count)
        self.post_trace = np.zeros(synapse_count)
        self.excitatory = ~self.snn.neurons.neuron_types[self.snn.synapses.source]

    def observe(self, spikes: Spikes):
        self.pre[:] = spikes.values[self.snn.synapses.source]
        self.post[:] = spikes.values[self.snn.synapses.target]
        self.eligibility *= self.trace_decay
        self.eligibility += self.pre_trace * self.post - self.post_trace * self.pre
        self.pre_trace = self.trace_decay * self.pre_trace + self.pre
        self.post_trace = self.trace_decay * self.post_trace + self.post

    def update(self, snn: SNN):
        if self.third_factor is not None:
            rpe = self.third_factor.calculate_rpe()
        else:
            rpe = 1.0

        if abs(rpe) < self.rpe_deadzone:
            return

        # Note: eligibility is *not* reset here. It already decays naturally
        # every tick in observe() (trace_decay), so this is safe - and
        # necessary - to call every tick: a continuous reward-modulated
        # (three-factor) update, rather than a single reward-modulated update
        # sampled at one instant while real reward-channel activity happens
        # continuously in between.
        snn.apply_weight_delta(self.learning_rate * rpe * self.eligibility * self.excitatory)
