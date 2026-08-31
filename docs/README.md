# Network Remake

This directory tracks the next-layer architecture that will be built around
the core synaptic neural network described in
[`NETWORK_ARCHITECTURE.md`](NETWORK_ARCHITECTURE.md).

## Boundary

The core SNN owns simulation concerns only:

- neuron and synapse state;
- compiled connectivity and weights;
- tick progression; and
- raw spike frames.

The layer documented here owns semantic interaction with the network:

- feature populations and their semantic values;
- `InputChannel` and `OutputChannel`;
- population encoders and detectors;
- event-oriented input and output; and
- temporal interpretation, arbitration, and host-facing protocols.

Channels may address individual features within a feature population, but a
channel is not created per feature. For example, one `OutputChannel` can
decode every character feature in `OUTPUT_CHARS`.

## Documents

- `NETWORK_ARCHITECTURE.md` — the core SNN and its compiled geometry;
- `NETWORK_ARCHITECTURE_SCALING.md` — a deferred locality-partitioning scaling
  extension;
- `INTERACTION_ARCHITECTURE.md` — input/output channels, feature semantics,
  encoders, detectors, and events;
- `CONVERSATION_ARCHITECTURE.md` — multi-turn exchanges and response
  completion;
- `SESSION_ARCHITECTURE.md` — running state, lifecycle, snapshots, and
  orchestration;
- `LEARNING_ARCHITECTURE.md` — activity observation and controlled weight
  updates;
- `MINIMAL_EXPERIMENT.md` — a concrete first experiment and its learning loop;
- `TRAINING_CONCEPT.md` — actor, critic, reward, RPE, and reward-modulated
  developmental training;
- `EVALUATION_REWARD_ARCHITECTURE.md` — outcome evaluation and reward signals;
- `CONTROL_ARCHITECTURE.md` — operational services, diagnostics, and runtime
  management.

These documents should describe layers around the SNN, not add semantic,
conversation, learning, or operational responsibilities to the SNN itself.
