# Network Architecture

This repository contains a small NumPy-based spiking conversational-agent
prototype. It is an experimental SNN, not a general language model or a claim
of full sequence learning. Character output uses the spiking/event architecture
described below; table-based language decoding is not supported.

In this document, **architecture** means behavior implemented and exercised by
the current code. `PLAN.md` uses **plan** for proposed or deferred work; planned
components are not part of the supported runtime API until implemented here.

## Current network

The simulator uses vectorised leaky integrate-and-fire neurons with sparse
synapses. Synaptic current is applied one tick after a spike. The network is a
single global recurrent graph, with an immutable, validated `PopulationLayout`
providing named populations and semantic subgroups.

The default agent layout is:

```text
INPUT          48
HIDDEN         48
AFFECT         7 x 4  = 28
OUTPUT_ACTION  7 x 4  = 28
OUTPUT_CHAR    12 x 3 = 36  (EOS plus alphabet)
total                     188 neurons
```

The hidden region is implicit recurrent space rather than a dedicated language
module. Explicit projections connect input to affect/actions, affect to
actions, and recurrent/input activity to character populations. The layout is
used by projection helpers, readouts, plasticity, and tests instead of relying
on duplicated offsets.

## Event and input protocol

`EventReadout` observes action or character population evidence.
It latches threshold crossings, applies a configurable cooldown, records
timestamps and inter-event intervals, and delegates simultaneous candidates to
deterministic arbitration. Silence emits no event; EOS is a discrete terminal
character event and stops that readout. A recurrent network is not reset
between output events.

Input presentation is bounded by `INPUT_BEGIN` and `INPUT_END`. The encoder
uses rate coding, and `presentation_speed` is the number of simulation ticks
for which each encoded frame is held. `presentation_speed > 1` preserves
the early-teaching slowdown; there is no separate duration override. Input
boundaries and response output are separate lifecycle concerns.

## Response lifecycle

`ResponseSession` validates these states and transitions:

```text
IDLE -> RECEIVING_INPUT -> RESPONDING -> EOS_RECEIVED -> COMPLETE
                                      \-> EXHAUSTED
```

`EXHAUSTED` records a response that reached its configured output limit without
EOS. Readout state is response-local by default; recurrent neuron state,
synaptic activity, affect, working memory, and plasticity eligibility persist
according to `SessionPolicy`.

## Training and reward

Ordered character events, including EOS, can be teacher-presented through the
existing recurrent network without feeding decoded characters back through
external `INPUT`. `evaluate_event_stream` accounts for correct, incorrect,
missing, valid-silence, unwanted, premature-EOS, missing-EOS, and post-EOS
outcomes with tolerant timing. Reports are retained in `reward_ledger`, and the
aggregate reward prediction error is committed to reward-modulated STDP after
the stream.

This is an initial recurrent-state and event-training baseline, not robust
general sequence memory or full sequence learning. Use
`train_response_events` with ordered character and EOS events for training.

## Main modules

- `src/continual_agent/simulation/neurons.py` — LIF neuron state and updates.
- `src/continual_agent/simulation/synapses.py` — sparse edges and weights.
- `src/continual_agent/simulation/network.py` — global recurrent simulation,
  spike history, and reset behaviour.
- `src/continual_agent/simulation/population_layout.py` — named populations,
  validated slices, and semantic subgroups.
- `src/continual_agent/plasticity/stdp.py` — eligibility-trace,
  reward-modulated STDP.
- `src/continual_agent/language/spiking_decoder.py` — character populations,
  teacher alignment, EOS handling, and generation helpers.
- `src/continual_agent/cognition/readout.py` — `EventReadout` and action
  arbitration.
- `src/continual_agent/encoding/text_encoder.py` — bounded rate-coded input
  presentation.
- `src/continual_agent/agent/session.py` — lifecycle, snapshots, and weight
  delta validation/merging.
- `src/continual_agent/agent/conversation_agent.py` — configuration,
  population construction, training, reward ledger, and orchestration.

## Session execution isolation

`ConversationAgent.execute_isolated` constructs an independent runtime shell,
copies the network's dynamic state and trainable weights, and runs response or
training callbacks against that shell. It records only net sparse `WeightDelta`
updates. Isolation defaults to `merge=False`; callers must opt in to applying
those deltas to the shared weight vector under an explicit conflict policy
(`REJECT`, `SUM`, or `AVERAGE`). Therefore isolated execution cannot mutate
shared neurons, recurrent activity, traces, readouts, affect, or working memory.
The callback's return value and the `SessionExecutionSnapshot` are the public
results; other mutable state produced inside the isolated shell is discarded.
Merging complete mutable agent state is outside this phase.

`SessionExecutionSnapshot.initial_synaptic_activity_state` keeps short-term
activity separate from long-term weights and supports weighted decayed
aggregation as a session baseline.

## Current limitations and deferred work

- There is no dedicated learned language/context-state population.
- Action selection is read directly from `OUTPUT_ACTION` through
  `EventReadout`; there is no parallel host-side policy.
- Plasticity has reward-modulated STDP, but no advanced schedules, structural
  rewiring, or active homeostasis/adaptive thresholds.
- `OUTPUT_FEEDBACK`, `OUTPUT_GATE`, advanced plasticity/homeostasis, and network
  scaling remain deferred.

## Recommended direction

1. Add explicit output feedback and evaluate it separately from output choice.
2. Evaluate whether a dedicated context population is justified by recurrent
   state measurements.
3. Add plasticity schedules and homeostasis before structural rewiring.
4. Increase network size only after event-stream tests and activity/capacity
   measurements demonstrate a genuine bottleneck.

## Useful verification

Focused tests cover population layout, output readout/arbitration, lifecycle and
isolation, input presentation, event-stream accounting, simulation, affect,
working memory, and vertical integration. Exact learned sequence behaviour
(`ma`, `ba`, `mama`, `baba`, repeated characters, and EOS) still needs stronger
dedicated tests; these are not evidence of full sequence learning.

Run:

```bash
python -m pytest -q
```
