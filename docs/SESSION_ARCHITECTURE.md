# Session Architecture

This document describes the running execution context around a `Conversation`.
Conversation semantics are defined in
[`CONVERSATION_ARCHITECTURE.md`](CONVERSATION_ARCHITECTURE.md); the SNN itself
is defined in [`NETWORK_ARCHITECTURE.md`](NETWORK_ARCHITECTURE.md).

## Boundary

The distinction is:

```text
Conversation = semantic turn history and protocol
Session      = live execution of that protocol
SNN          = neural state transition
```

A Session owns a live SNN and executes one Conversation across one or more
turns. It coordinates the layers around the SNN, but does not absorb their
logic.

Session is also the owner of tick orchestration. Stateful services such as
background drive, homeostasis, metrics, and Learning may be registered with
the Session as pre-step producers or post-step observers. The services own
their internal state; Session owns their participation order.

For now, Session does not need a general drive-merging abstraction. It builds
one local drive per population from the applicable Channel and control inputs.
Most populations should have only one contributor initially; homeostatic drive
is the expected early overlap. If contributors do overlap, Session sums them
and clamps the result before passing it to the SNN.

## Current responsibilities

Session should initially own:

- executing Conversation turns;
- scheduling `InputChannel` and `OutputChannel` activity around SNN ticks;
- invoking registered pre-step drive producers and post-step observers;
- passing population-local input drives to the SNN;
- forwarding spike frames to output consumers;
- enforcing input and response boundaries;
- handling EOS, response timeout, cancellation, and failure; and
- keeping one running execution alive across multiple turns.

When training is enabled, Session also coordinates the learning boundary:

```text
SNN ticks → Learning records spikes and eligibility
turn completion → Evaluation produces outcome
outcome → reward policy produces reward
reward + critic values → Learning calculates RPE and updates weights
```

The update should occur once for the accounted evaluation, at an explicit
turn/training-step boundary. Session coordinates ordering and lifecycle; it does
not calculate semantic reward or implement STDP.

The SNN remains the authority for neural time. Session must not maintain a
second independent tick counter; it uses the tick carried by returned `Spikes`.

Session also should not decide what a feature means, how a population is
encoded, how output evidence is detected, or whether a response is correct.
Those responsibilities belong to Interaction and Evaluation respectively.

## Turn execution

A typical turn is coordinated like this:

```text
Conversation opens input
    → Session schedules input feature activations
    → Channels and control services produce population-local drives
    → overlapping drives are summed and clamped
    → Session passes the population-to-drive map to SNN.step()
    → SNN returns Spikes
    → Channels and post-step services observe the frame
    → Conversation records semantic output events
    → EOS, timeout, cancellation, or failure completes the turn
```

Input and output activity may interleave. Completing one turn does not require
creating a new SNN; the Session decides which state is allowed to continue
into the next turn once those policies have been defined.

## Deferred state policies

The following are intentionally TODOs until the synthetic temporal tests give
us stable behavioral requirements:

- define snapshot contents and restore semantics;
- define reset policies between ticks, turns, conversations, and sessions;
- define persistence for neural, channel, detector, and learning state;
- define isolation, cloning, and weight-sharing behavior; and
- define lifecycle guarantees after cancellation or failure.

These policies must eventually account for every mutable component, including
SNN state, weights, pending current, Channel/Detector state, plasticity,
background drive, homeostasis, metrics, and ablation state. Until then, the
Session boundary is established without promising a final reset or snapshot
model.

## Future layers

Learning will attach to the SNN through its controlled weight-update seam, but
Session should only coordinate when learning observes or applies updates.
Evaluation will consume completed turns or Conversation history and should not
be embedded in Session.

Operational concerns such as diagnostics, homeostasis, ablation, and runtime
management will eventually be described by the Control architecture.
