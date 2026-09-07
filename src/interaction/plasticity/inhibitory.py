"""Inhibitory synaptic plasticity (Vogels et al. 2011) - inhibitory
synapses only, no reward signal at all. Potentiates (strengthens
inhibition) on correlated pre/post activity in *either* spike order, unlike
excitatory STDP's causally-asymmetric window, balanced against a constant
depression term tied to `target_rate` - net effect, inhibitory synapses
onto a neuron grow until that neuron's firing rate settles near the
target. A real, local, reward-independent rule - not a general-purpose
homeostat for the rest of the network, just this one plasticity mechanism."""

from dataclasses import dataclass, field

import numpy as np

from ...network.snn import SNN, Spikes
from ..plugins import Observer


@dataclass
class InhibitoryPlasticity(Observer):
    snn: SNN
    # Desired average firing probability per tick for postsynaptic neurons -
    # derive this from the network's own measured baseline activity rather
    # than picking one (see build_*_experiment callers).
    target_rate: float
    learning_rate: float = 0.001
    trace_decay: float = 0.95

    # TODO: pre_trace/post_trace are identical to Hebbian's - extract and share.
    pre_trace: np.ndarray = field(init=False)
    post_trace: np.ndarray = field(init=False)
    inhibitory: np.ndarray = field(init=False)
    depression: float = field(init=False)

    def __post_init__(self):
        synapse_count = len(self.snn.synapses.source)
        self.pre_trace = np.zeros(synapse_count)
        self.post_trace = np.zeros(synapse_count)
        self.inhibitory = self.snn.neurons.neuron_types[self.snn.synapses.source]
        self.depression = 2.0 * self.target_rate / (1.0 - self.trace_decay)

    def observe(self, spikes: Spikes):
        source_spiked = spikes.values[self.snn.synapses.source]
        target_spiked = spikes.values[self.snn.synapses.target]

        delta = np.zeros_like(self.pre_trace)
        pre_spike = source_spiked & self.inhibitory
        post_spike = target_spiked & self.inhibitory
        delta[pre_spike] += self.post_trace[pre_spike] - self.depression
        delta[post_spike] += self.pre_trace[post_spike]

        # Potentiation (positive delta) makes inhibition stronger, i.e. the
        # weight more negative.
        self.snn.apply_weight_delta(-self.learning_rate * delta)

        self.pre_trace = self.trace_decay * self.pre_trace + source_spiked
        self.post_trace = self.trace_decay * self.post_trace + target_spiked
