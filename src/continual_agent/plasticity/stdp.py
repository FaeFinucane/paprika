"""Eligibility-trace STDP with a delayed third factor."""

from __future__ import annotations

import numpy as np

from continual_agent.simulation.synapses import SparseSynapses


class RewardModulatedSTDP:
    """Pair-based STDP whose eligibility is committed by reward.

    The spike timing rule is local: each edge only reads the source and target
    spike traces. Reward is deliberately supplied separately at the end of an
    interaction, making delayed credit visible and testable.
    """

    def __init__(
        self,
        synapses: SparseSynapses,
        dt: float = 1.0,
        trace_tau: float = 8.0,
        eligibility_tau: float = 30.0,
        learning_rate: float = 0.01,
        potentiation: float = 1.0,
        depression: float = 0.75,
        weight_limit: float = 1.0,
    ):
        self.synapses = synapses
        self.decay_trace = np.exp(-dt / trace_tau)
        self.decay_eligibility = np.exp(-dt / eligibility_tau)
        self.learning_rate = learning_rate
        self.potentiation = potentiation
        self.depression = depression
        self.weight_limit = weight_limit
        self.pre_trace = np.zeros(synapses.neuron_count)
        self.post_trace = np.zeros(synapses.neuron_count)
        self.eligibility = np.zeros(synapses.weight.size)

    def observe(self, spikes: np.ndarray) -> None:
        spikes = np.asarray(spikes, dtype=float)
        if spikes.shape != (self.synapses.neuron_count,):
            raise ValueError("spikes have the wrong shape")

        self.pre_trace *= self.decay_trace
        self.post_trace *= self.decay_trace
        self.eligibility *= self.decay_eligibility

        pre_spike = spikes[self.synapses.source]
        post_spike = spikes[self.synapses.target]
        self.eligibility += (
            self.potentiation * post_spike * self.pre_trace[self.synapses.source]
            - self.depression * pre_spike * self.post_trace[self.synapses.target]
        )
        self.pre_trace += spikes
        self.post_trace += spikes

    def reinforce(
        self,
        reward_prediction_error: float,
        modulation: float = 1.0,
        target_neurons: np.ndarray | None = None,
    ) -> None:
        update = self.learning_rate * reward_prediction_error * modulation * self.eligibility
        if target_neurons is not None:
            target_mask = np.zeros(self.synapses.neuron_count, dtype=bool)
            target_mask[np.asarray(target_neurons, dtype=int)] = True
            update = np.where(target_mask[self.synapses.target], update, 0.0)
        self.synapses.weight[:] = np.clip(
            self.synapses.weight + update,
            -self.weight_limit,
            self.weight_limit,
        )

    def reset_traces(self) -> None:
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.eligibility.fill(0.0)
