# Learning Architecture

Learning is a sidecar to the core SNN. The SNN owns neurons, synapses, and
weights; Learning observes activity and proposes or applies controlled weight
updates. Learning does not become part of Channels or Conversation.

The initial training concept uses one SNN with distinct functional populations:

```text
shared state → actor/output populations
             → critic/value populations
             → reward-system populations
```

These are roles within one network, not necessarily separate network objects.
Their synapses may all use the same reward-modulated plasticity mechanism, while
their connectivity and local eligibility traces determine what each population
learns.

## Flow

```text
SNN step → spike frame → Learning observes activity
                              ↓
                     traces / eligibility / state
                               ↓
                  reward or teaching signal arrives
                               ↓
                 RPE = r + gamma × V_next − V_current
                               ↓
              reward-modulated updates to eligible synapses
```

Learning may support several policies over time, including local STDP,
reward-modulated plasticity, and supervised teaching. These policies should
share a narrow attachment point rather than being embedded in the SNN.

## Responsibilities

Learning should own:

- activity traces and eligibility state;
- learning-rule parameters;
- consuming reward, RPE, or supervised signals;
- calculating bounded weight updates; and
- learning diagnostics.

For reward-modulated plasticity, Learning should distinguish three signals:

```text
reward system produces current reward, r
critic predicts future reward, V
RPE compares the outcome with the prediction
```

The RPE may be delivered as a signed, population-level modulatory signal. A
single network-wide modulation can update eligible actor, critic, reward-system,
and shared-representation synapses. It must not update every recently active
synapse: local spike timing, trace decay, and circuit/population gating provide
the approximate credit assignment.

Eligibility must survive the delay between an output and its evaluation, but it
must decay so unrelated earlier activity is not reinforced. Waiting for an
evaluation is not itself a reward of zero or a punishment; reward omission is
negative only at a boundary where reward was expected.

The core SNN should own the actual synaptic weights and expose a controlled
update operation. Updates must respect the compiled graph, weight limits, and
any active ablation or pathway constraints.

The architecture must define when updates become visible. A simple initial rule
is to observe the spike frame after a tick and apply updates at an explicit
turn or training-step boundary, rather than mutating weights halfway through a
neural transition.

## Interaction with other layers

- Session coordinates when learning observes activity or applies updates.
- Evaluation and reward may provide a learning signal, but are not required.
- Conversation provides semantic context but does not update weights.
- Control may enable, disable, inspect, or schedule learning.

## Deferred decisions

- the exact SNN weight-update API;
- per-tick versus end-of-turn update timing;
- supervised teaching versus plasticity composition;
- actor/critic/reward population layout and readout conventions;
- reward-modulated STDP eligibility and delay parameters;
- how internal and external reward are combined during development;
- weight-update transactions and concurrency;
- snapshot and reset requirements for traces; and
- how learning state participates in isolated execution.
