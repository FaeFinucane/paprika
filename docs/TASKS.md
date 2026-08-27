# Tasks

This file contains deferred or actionable work, ordered by priority. Implemented
phases and historical cleanup decisions are intentionally not listed as active
tasks.

## Priority: experiment scaling

- Add an optional held-out evaluation split for larger or overtraining-prone
  experiments. The tiny synthetic experiment may continue to use its training
  sequences while it is a smoke test, but larger runs should report training
  and held-out metrics separately, including per-sequence results.

## Priority: biologically honest projection seeding

- Replace the current feature-partitioned INPUT→HIDDEN bootstrap and dense
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

## Neural dynamics and output protocol

- Sweep the synthetic membrane time constant explicitly, including a candidate
  around `tau=10` ticks, together with frame spacing and response duration.
  `tau=3` is a historical responsiveness heuristic, not a calibrated biological
  parameter.
- Add and evaluate a separate `OUTPUT_FEEDBACK` channel only after the event
  protocol has stable coverage. It must represent self-generated sensory
  reafference, not silently become a second decision-maker or external input.
- Add an `OUTPUT_GATE` only as post-selection emission control; it must not
  alter event selection.

New external stimulation should implement the `Drive` protocol; diagnostics,
homeostasis, and learning observations should implement a scheduled
`NetworkPlugin`.

See [architecture](ARCHITECTURE.md) and [training](TRAINING.md) for the current
implemented boundary.
