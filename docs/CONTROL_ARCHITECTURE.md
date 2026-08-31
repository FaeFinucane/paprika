# Control Architecture

Control is the operational layer around the SNN, interaction, session,
learning, and evaluation layers. It provides stateful services that participate
in SNN ticks without owning semantic conversation behavior or neuron dynamics.

## Responsibilities

Control may eventually own:

- diagnostics and runtime metrics;
- background drive;
- population homeostasis;
- internal reward and drive sources;
- ablation and pathway controls;
- checkpoint and restore orchestration;
- service configuration and scheduling; and
- locality-partition or device management.

These services may observe spikes, voltage, weights, and session state, but
they should use explicit APIs rather than reaching into arbitrary mutable
arrays. A service owns its own configuration and runtime state; Session owns
when that service participates in execution.

## Drive and state boundaries

Background and homeostatic drive are operational inputs composed with
Channel-produced population drives before the SNN advances. They are not
semantic feature writes. All of them use the same population-to-local-drive
shape accepted by `SNN.step()`. Where drives overlap, the current rule is to
sum them and clamp the result; a dedicated drive-composition abstraction is
not needed yet.

The basic tick flow is:

```text
pre-step services produce drive
    → Session combines Channel and control drives
    → SNN.step()
    → post-step services observe Spikes and state
    → stateful services prepare the next tick
```

Background drive is a pre-step producer. It may own an RNG or another evolving
drive state. Homeostasis is a feedback controller: it observes completed
spikes, updates its population-level drive, and affects a later tick rather
than mutating the current neural transition halfway through. Metrics are
observers and should not affect execution unless explicitly used by a control
service such as homeostasis.

Internal reward sources follow the same boundary. They may observe internal
state, population activity, or completed outcomes and produce reward input for a
later evaluation/training boundary. They should not bypass Channels or mutate
weights directly. A homeostatic variable can provide a primitive reward by
becoming more or less satisfied; Learning can then associate that reward with
recent eligible activity.

Ablation changes which compiled pathways are active; it must not silently
change the identity of the compiled network. Metrics observe execution and
must not influence it except through an explicitly configured control service
such as homeostasis.

## Coordination

Control can configure and supervise other layers:

```text
Control
  ├── Session / runtime lifecycle
  ├── SNN and Learning configuration
  ├── diagnostics and metrics
  └── homeostasis, background drive, and ablation
```

It should not become a generic plugin container or a catch-all for policies
that belong to Conversation, Evaluation, or Learning.

Session should register the control services that participate in a running
execution, so it can invoke them in a deterministic order and eventually
include their state in lifecycle operations. This registration is a runtime
coordination boundary, not a requirement that every service share one base
class.

## Deferred decisions

- which services are enabled by default;
- service ordering around each SNN tick;
- how internal reward sources are combined and bounded;
- control-state snapshot and reset behavior;
- thread safety and concurrent control requests;
- distributed ownership of operational state; and
- thresholds at which locality partitioning becomes worthwhile.
