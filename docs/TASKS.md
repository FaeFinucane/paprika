# Tasks

This file contains deferred or actionable work, ordered by priority. Implemented
phases and historical cleanup decisions are intentionally not listed as active
tasks. The first priorities are experiment-validity fixes: otherwise the neural
results are liable to measure protocol artefacts rather than learning.

## Priority: experimental validity and measurable learning

- Add an optional held-out evaluation split for larger or overtraining-prone
  experiments. The tiny synthetic experiment may continue to use its training
  sequences while it is a smoke test, but larger runs must report training and
  held-out metrics separately, including per-sequence results.
- Make trial state policy explicit. Add configuration for preserving or
  resetting membrane, refractory, pending-current, recurrent, readout, affect,
  working-memory, eligibility, and background-drive RNG state. Compare
  stateful-lifetime and cold-start evaluation without treating cold-start as
  the only valid intelligence measure. Ensure evaluation does not accidentally
  inherit state when a controlled comparison is requested.
- Categorize synthetic-temporal tests into protocol/unit tests and behavioral
  architecture tests. Keep mechanics tests for boundaries, readout, and event
  accounting; add a separate behavioral harness that requires event
  precision/recall, exact or partial sequence accuracy, EOS accuracy, latency,
  and comparison with untrained/no-learning controls. Do not treat weight
  movement as evidence of temporal copying.
- Add dedicated event-stream cases for `ma`, `ba`, `mama`, `baba`, repeated
  characters, valid silence, premature EOS, missing EOS, and post-EOS output.
  Clarify and test inclusive/exclusive `end_time` and silence-boundary
  semantics.

## Priority: lifecycle and learning semantics

- Centralize reward-to-RPE conversion and baseline updates across conversational
  and synthetic training. Prefer one event-based commitment path for ordinary
  observed events; if missing/deferred outcomes are settled at trial end, make
  that explicit and prevent duplicate application of event rewards.
- Keep diffuse reinforcement as the current deliberate baseline while hidden
  population roles are unknown. Add pathway, output, or activity-derived
  assembly masks as comparison conditions rather than assuming specific hidden
  neurons in advance. Log signed and absolute updates, clipping, and saturation.
- Document and evaluate the early reward schedule's positive reward for
  incorrect and unwanted events as exploration bootstrapping. Compare neutral
  and negative alternatives when scaling experiments, and always report event
  quality separately from aggregate reward.
- Decide whether supervised delayed-copy training is an explicitly named
  structural/oracle baseline or a spike-based learning method. Add a
  free-running evaluation with teacher signals removed.

## Priority: runtime correctness and API invariants

- Make connectivity immutable after construction, or add an atomic mutation API
  that resizes STDP eligibility, ablation masks, and every other edge-indexed
  structure when edges are added.
- Validate `NetworkConfig` completely: positive dimensions, unique output
  tokens and subgroup names, required `<EOS>`, valid population coverage, and
  non-empty protocol groups.
- Implement `persist_affect` and `persist_working_memory` semantics fully, or
  remove those options from the runtime policy and make ownership explicit.
- Return actual elapsed simulation ticks from response APIs. Keep event count,
  latency, and duration as separate metrics.

## Priority: test and observability quality

- Repair the affect causality test to compare intact and ablated cloned
  snapshots, inspect only the intended population, and assert a behavioral or
  activity effect rather than merely a voltage difference.
- Strengthen the curriculum test so it checks expected action accuracy, affect
  targets, reward improvement, and an untrained baseline; otherwise rename it
  to describe its current smoke-test behavior.
- Test that working memory changes decisions, outputs, or learning—not only
  that internal state persists.
- Add decoder correctness tests for known spike frames, competing tokens,
  repeated characters, EOS truncation, unknown tokens, and free-running output.
- Add direct tests for `UserMessage`, `Feedback`, and `AgentAction`, including
  validation, boundary values, serialization if supported, and integration with
  environment APIs.
- Test core simulation validation, reset/snapshot behavior, pending-current
  semantics, malformed synapse arrays, duplicate edges, and homeostasis
  integration effects.
- Distinguish public-contract tests from white-box tests that depend on private
  methods or direct weight mutation.
- Rename metrics whose names do not match their calculations, especially
  `hidden_activity` and `voltage_spread`; expose episode-average firing,
  cross-neuron spread, temporal variance, integrated eligibility, and signed
  pathway deltas separately.

## Priority: simplification and maintainability

- Replace projection-index arithmetic and combined-edge reinterpretation with
  named projection builders that return explicit edge ranges or index arrays.
- Introduce small typing protocols for runtime services, task adapters, readouts,
  and facade mixins instead of relying on `Any`.
- Make affect evaluation isolated and deterministic by default; validate target
  fields, ranges, and coverage.
- Decide and document the public API surface consistently. Export the canonical
  facade and environment/evaluation boundary types, or explicitly document
  that they are subpackage-only.
- Align or deliberately document the differing feature geometries used by
  temporal text encoding and `feature_vector()`, so transfer between response,
  affect, and token training is interpretable.
- Restore configured affect initial values on reset, or document that reset
  always returns to fixed neutral baselines.
- Validate shapes at the `AffectiveCircuit` boundary rather than relying on
  later NumPy indexing errors.

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
  choose an action or character.

Task adapters own encoding, session boundaries, readout, and reward semantics.
They compose the shared runtime and must not construct or step a second network.
New external stimulation should implement the `Drive` protocol; diagnostics,
homeostasis, and learning observations should implement a scheduled
`NetworkPlugin`.

See [architecture](ARCHITECTURE.md) and [training](TRAINING.md) for the current
implemented boundary.
