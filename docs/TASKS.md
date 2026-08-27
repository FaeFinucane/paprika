# Tasks

This file contains deferred or actionable work. Implemented phases and historical
cleanup decisions are intentionally not listed as active tasks.

## Priority: biologically honest projection seeding

- Replace the current feature-partitioned input→hidden bootstrap and dense
  hidden-group recurrence with a shared sparse projection model. Each source
  neuron should sample an overlapping target subset controlled by an expected
  `fan` value; A and B must not be forced into disjoint hidden subsets.
- Use a bounded bimodal weight distribution with a strong positive mode and a
  weaker negative mode, exposing inhibitory fraction and both mode
  distributions in `WeightInitializationConfig`.
- Use the same projection style for INPUT→HIDDEN and HIDDEN→HIDDEN while
  allowing separate fan/distribution settings. Avoid duplicate contacts and
  preserve deterministic named RNG streams.
- Replace supervised delayed-copy assumptions about fixed hidden feature groups
  with activity-derived temporary hidden assemblies. Measure whether the
  resulting overlapping recurrent network can retain symbols before using it
  as an STDP substrate.

## Output protocol

- Add and evaluate a separate `OUTPUT_FEEDBACK` channel only after the event
  protocol has stable coverage. It must not reuse external `INPUT` or become a
  second decision-maker.
- When output feedback is revisited, consider it as self-generated sensory
  reafference rather than automatic token autoregression: an emitted event may
  produce a separate internal sensory signal that can contribute to prediction
  error and affective modulation.
- Add an `OUTPUT_GATE` only as a post-selection emission control; it must not
  choose an action or character.

## Learning and stability

- Sweep the synthetic membrane time constant explicitly (including a more
  brain-like candidate around `tau=10` ticks) together with frame spacing and
  response duration; `tau=3` is a historical responsiveness heuristic, not a
  calibrated biological parameter.
- Add stronger dedicated event-stream tests for `ma`, `ba`, `mama`, `baba`,
  repeated characters, valid silence, premature EOS, missing EOS, and post-EOS
  suppression.
- Compare stateful-lifetime evaluation (continuing membrane, refractory,
  pending-current, and recurrent state) with explicit cold-start evaluation;
  report training and evaluation metric windows separately.

See [architecture](ARCHITECTURE.md) and [training](TRAINING.md) for the current
implemented boundary.

Task adapters own encoding, session boundaries, readout, and reward semantics.
They compose the shared runtime and must not construct or step a second network.
New external stimulation should implement the `Drive` protocol; diagnostics,
homeostasis, and learning observations should implement a scheduled
`NetworkPlugin`.
