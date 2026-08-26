# Network Evolution Plan

This plan describes the continuously active spiking conversational agent. It
is a small experimental SNN, not yet a general language model or a full
sequence-learning system.

This file describes proposed evolution and deferred experiments, not additional
runtime APIs. The implemented architecture and supported terminology are
defined in `ARCHITECTURE.md`.

## Implemented baseline

### Phase 1: Named population layout

`PopulationLayout` provides immutable, validated slices for `INPUT`, `HIDDEN`,
`AFFECT`, `OUTPUT_ACTION`, and `OUTPUT_CHAR`, including semantic subgroups for
affect signals, actions, and characters. Projection helpers, readouts,
plasticity, and tests use the layout rather than duplicated offsets.

### Phase 2: Lifecycle and isolated sessions

`ResponseSession` validates the states:

```text
IDLE -> RECEIVING_INPUT -> RESPONDING -> EOS_RECEIVED -> COMPLETE
                                      \-> EXHAUSTED
```

`EXHAUSTED` is used when the configured response limit is reached without EOS.
Response-local readout state is reset at a new response, while recurrent state,
synaptic activity, affect, working memory, and eligibility persist according to
`SessionPolicy`.

`execute_isolated` runs against copied neural state, traces, readouts, affect,
memory, and weights. It returns the callback result plus a session execution
snapshot containing validated sparse `WeightDelta` records. It defaults to no
merge; callers must explicitly opt in to merging weight changes using `REJECT`,
`SUM`, or `AVERAGE`. Whole mutable agent state is never merged, and other state
produced by the isolated callback is discarded. Initial synaptic activity has a
separate weighted/decayed aggregation path from long-term weights.

### Phase 3: Continuous event readout

`EventReadout` observes output-population spikes or activation evidence on each
tick. It emits timestamped events, distinguishes silence from EOS, prevents
duplicates with latching and cooldown, records inter-event intervals, and
stops after EOS without resetting recurrent state between events.

### Phase 4: Output arbitration

`OutputArbitrationPolicy` selects at most one candidate deterministically.
Evidence wins first; EOS, action, character, and stable subgroup priorities
resolve ties. Arbitration decisions are observable and separate from any
future output gate.

### Phase 5: Input presentation

`TextEncoder.present` wraps rate-coded input in explicit `INPUT_BEGIN` and
`INPUT_END` signals. Its single `presentation_speed` setting holds each encoded
frame for that many simulation ticks. Values above one provide the early
teaching slowdown; normal speed is one.

### Phase 7 baseline: Recurrent state validation

The existing hidden/recurrent network is used as internal state; no speculative
`LANGUAGE_STATE` population has been added. Teacher-guided ordered character
events train recurrent transitions without feeding decoded characters back
through external `INPUT`. State persistence and the host-side `WorkingMemory`
scaffold can be compared, but this remains an initial validation baseline, not
robust general sequence memory or full sequence learning.

### Phase 8 baseline: Event reward and training

Targets are ordered events, including EOS. `evaluate_event_stream` uses tolerant
timing and accounts for correct, incorrect, missing, valid-silence, unwanted,
premature-EOS, missing-EOS, and post-EOS output. `ConversationAgent` retains
the resulting records in `reward_ledger` and commits the aggregate reward
prediction error to reward-modulated STDP after eligibility has accumulated.

## Deferred phases

### Phase 6: `OUTPUT_FEEDBACK`

Deferred. Add a separate internal confirmation channel only after the baseline
event protocol is stable. It must not reuse external `INPUT` or become a second
decision-maker.

### Phase 10: `OUTPUT_GATE`

Deferred. A future gate may control whether an already selected candidate is
externally emitted; it must not select the character or action.

### Phase 9: Advanced plasticity and homeostasis

Deferred. Reward-modulated STDP is present, but configurable plasticity
schedules, structural rewiring, adaptive thresholds, and other homeostatic
stabilisation are not implemented.

### Phase 11: Scaling

Deferred. The default network has approximately 188 neurons:

```text
INPUT 48, HIDDEN 48, AFFECT 28, OUTPUT_ACTION 28, OUTPUT_CHAR 36
```

Scale only after event-stream behavior, recurrent-state measurements, and
activity/capacity tests demonstrate a genuine bottleneck. Do not treat longer
sequence examples alone as evidence that scaling is needed.

## Verification principles

Keep focused coverage for layout, lifecycle, isolation and delta merging,
input presentation, event readout/arbitration, event accounting, simulation,
plasticity, affect, memory, and vertical integration. Evaluate neural behavior
through the canonical output event readout.

Exact learned behavior for examples such as `ma`, `ba`, `mama`, `baba`, repeated
characters, valid silence, premature EOS, and post-EOS suppression still needs
stronger dedicated tests; no current result should be described as full
sequence learning.

The following remain explicitly deferred until the baseline is observable:

- `OUTPUT_FEEDBACK`;
- `OUTPUT_GATE`;
- advanced plasticity, structural rewiring, and homeostasis;
- network scaling.
