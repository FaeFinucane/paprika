# Synthetic Temporal Experiment

The canonical paired run is `python -m continual_agent.experiments.synthetic_temporal`.
It compares a fresh supervised structural baseline with a fresh reward-modulated
STDP agent for every condition and control. `run_temporal_experiment` remains
available for focused single-agent tests; `run_paired_temporal_experiment` selects
both agents.

The temporal copy experiment is a non-linguistic workload. It injects small
numeric frames directly through `INPUT_BEGIN` and `INPUT_END`; it does not use
`TextEncoder`, text normalization, English tokens, or decoded-character
feedback.

## Architecture

`synthetic_temporal.py` only generates frames, selects fresh-runtime conditions,
and evaluates timestamped output events. Each trial constructs a normal
`SpikingRuntime` with a synthetic output alphabet. Network populations,
synapses, `EventReadout` arbitration, session boundaries, and teacher alignment
are the production implementations. `SpikingRuntime.train_input_events` and
`run_input_events` are the generic raw-frame interfaces used by the experiment.

The immediate condition permits readout during the input window. Delayed copy
withholds readout until `INPUT_END`, then scores the complete `A -> EOS` or
`B -> EOS` target with a generous patience window. No host-side queue is supplied.
The supervised baseline updates only hidden-to-output structural weights; the
direct input-to-output projection is therefore a meaningful ablation. STDP
training presents a teaching pulse but changes weights only when its delayed
scalar reward is committed through `RewardModulatedSTDP.reinforce`.

## Tests

Mechanics tests cover `EventReadout` arbitration, silence and EOS semantics,
event-stream accounting, and input-session protocol errors. Those tests do not
claim that a neural network can solve temporal copy.

The temporal experiment tests cover production-path viability: fresh agents,
immediate versus delayed timing, training/no-learning/untrained separation,
target shuffling, and explicit pathway controls. They are not a second neural
agent implementation.

## Controls

The direct input-to-character projection and hidden-to-character recurrent
projection are exposed by the production runtime and can be ablated per trial.
The runtime's persistent edge mask is applied after every network and teacher
update, so an ablated edge remains disabled throughout training and evaluation.
The delayed condition is a retention probe, not a claim of general sequence
memory. A delayed symbol that survives direct-path ablation is evidence that the
output used the hidden-mediated route; it is not evidence of language understanding.
Untrained and no-learning cells measure initialization and protocol effects,
while recurrent and direct-path ablations identify which projections carry the
observed event. Timing is measured, not treated as an exact target.
