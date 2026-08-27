"""Eligibility-trace STDP with a delayed third factor."""

from __future__ import annotations

import numpy as np

from continual_agent.simulation.synapses import SYNAPTIC_WEIGHT_LIMIT, SparseSynapses


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
        weight_limit: float = SYNAPTIC_WEIGHT_LIMIT,
    ):
        self.synapses = synapses
        self.decay_trace = np.exp(-dt / trace_tau)
        self.decay_eligibility = np.exp(-dt / eligibility_tau)
        self.learning_rate = learning_rate
        self.potentiation = potentiation
        self.depression = depression
        if weight_limit <= 0.0:
            raise ValueError("weight_limit must be positive")
        self.weight_limit = min(weight_limit, SYNAPTIC_WEIGHT_LIMIT)
        self.pre_trace = np.zeros(synapses.neuron_count)
        self.post_trace = np.zeros(synapses.neuron_count)
        self.eligibility = np.zeros(synapses.weight.size)
        self.last_update = np.zeros(synapses.weight.size)
        self.total_signed_update = 0.0
        self.total_absolute_update = 0.0
        self.total_clipped_update = 0.0

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
        before = self.synapses.weight.copy()
        proposed = before + update
        clipped = np.clip(
            proposed,
            -self.weight_limit,
            self.weight_limit,
        )
        self.synapses.weight[:] = clipped
        self.last_update[:] = clipped - before
        self.total_signed_update += float(self.last_update.sum())
        self.total_absolute_update += float(np.abs(self.last_update).sum())
        self.total_clipped_update += float(np.abs(proposed - clipped).sum())

    def reset_traces(self) -> None:
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.eligibility.fill(0.0)
