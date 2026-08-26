# Clean-up Pass Plan

The goal is a small, coherent research codebase rather than a museum of
superseded APIs. Every source file, public symbol, configuration option, and
test should have a current architectural or experimental justification.

This pass should remove compatibility wrappers and duplicate implementations,
not preserve them “just in case”. Existing behavior should only be retained
when it is part of the supported architecture or a deliberately named,
currently useful experiment.

## Decisions to establish first

Before deleting APIs, confirm the supported contracts:

- `PopulationLayout` is the only population-addressing API. Raw starts and
  public offset fields are removed, not wrapped.
- `EventReadout` is the sole externally visible output-event mechanism.
- `SpikingCharacterDecoder` is the sole language-output implementation.
- `AffectiveState` is the sole affect state object; `DriveState` is obsolete.
- Isolated session execution defaults to no merge; weight changes are merged
  only through an explicit operation.
- Input presentation has one supported semantic mode: rate-coded input with
  configurable presentation speed. Pulse presentation is confirmed as a
  historical hold-over and must be removed completely.
- Host-side action policy weights are confirmed as an unneeded parallel
  mechanism and must be removed from the production agent. No separate host
  policy should select actions or be copied into neural synapses.

## Phase 1: Inventory and define the supported surface

Create a symbol-level inventory before editing. For every class, function,
field, configuration option, export, and test, record one of:

1. required by the current architecture;
2. required by a named active experiment;
3. test/support infrastructure with a clear owner;
4. obsolete and removable;
5. ambiguous and requiring a decision.

Use code search and test collection to verify callers. Do not treat a symbol's
presence in an `__init__.py` as justification for keeping it.

## Phase 2: Remove obsolete parallel implementations

Delete the table-based language path if it is not an active experiment:

- `language/decoder.py` and `CharacterDecoder`;
- `experiments/learn_english.py`;
- `tests/test_decoder.py`;
- stale package exports.

Keep only the spiking language path and rewrite documentation/tests to describe
it as canonical. If the experiment is valuable, move it to an explicitly
archival/experimental area with a clear non-production status instead of
pretending it is a second supported implementation.

## Phase 3: Enforce population-only addressing

Remove integer-offset compatibility paths from:

- `SpikingCharacterDecoder` grouping and projection helpers;
- `ActionReadout` named grouping data;
- `AffectiveCircuit` grouping/projection helpers;
- `ConversationAgent` public offset fields;
- tests that exist solely to preserve these paths.

All callers should use `PopulationLayout` and named subgroup selection. Remove
duplicated layout construction, hard-coded `DEFAULT_LAYOUT` starts, unused
locals, and any helper whose purpose is merely to translate an old offset API.

Add negative tests for invalid population ownership/bounds rather than tests
for legacy compatibility.

## Phase 4: Consolidate output and training APIs

Make the event path canonical:

- remove fixed-window `output_score`, `choose`, and `choose_non_eos` helpers;
- remove `ContinuousOutputReadout` alias;
- remove unused `ActionReadout.decide()` if no current training/evaluation path
  requires it;
- remove `ActionReadout.policy_weights`, `score_policy()`, and
  `reinforce_policy()`; action decisions must come from
  `OUTPUT_ACTION` → `EventReadout`;
- remove policy-weight copying and duplicate direct action updates from
  `ConversationAgent`; retain only the chosen neural training mechanism;
- remove `exploration_epsilon` unless exploration remains an explicitly
  supported part of that neural training mechanism;
- remove `train_response_text()` and update curriculum/tests to call the
  event-stream API directly;
- remove `respond_with_working_memory()`.

The resulting training API must have one clear owner and one meaning for each
operation. Do not retain wrappers whose only purpose is old naming. If a host
policy baseline is ever useful, put it in a separate explicitly named
experiment; do not embed it in `ConversationAgent` or copy its weights into
the neural network.

## Phase 5: Simplify input presentation

Use one supported input protocol and make its semantics explicit:

- `INPUT_BEGIN` and `INPUT_END` remain protocol boundaries;
- presentation speed is a curriculum/training parameter, not a second input
  meaning;
- remove unused `InputEvent.semantic_index` and dead encode allocations;
- use only canonical rate-coded presentation, without alternate presentation modes;
- remove `input_presentation_rate` or equivalent settings if they are not
  consumed by the canonical rate encoder;
- retain only `presentation_speed`, measured as ticks per encoded frame. Values
  above one slow early teaching; one restores normal speed.

## Phase 6: Simplify state, policies, and session APIs

Remove aliases and policy options without a current use:

- delete `ConversationAgent.drives` and the obsolete `DriveState` module/export;
- remove `ConflictPolicy.ADD` in favour of one canonical spelling, likely
  `SUM`;
- remove duplicate input protocol constant aliases;
- simplify `SessionPolicy` fields that are not independently configurable or
  observed;
- ensure isolated execution returns the useful result/state it claims to
  return, while discarded state is not represented as a supported API;
- keep conflict policy only if parallel session merging is an active feature;
  otherwise defer or remove it with its tests.

Avoid abstraction for abstraction's sake. A policy object stays only when it
represents a meaningful user-controlled decision or a separately tested
invariant.

## Phase 7: Prune dead state and configuration

Review and either remove or explicitly wire up:

- unused affect recovery methods;
- duplicate synaptic activity snapshot APIs;
- hard-coded/default layouts that can disagree with configured alphabets;
- unused configuration fields and stale comments;
- affective-state specification sections describing unimplemented guarantees.

Do not delete a behavior merely because it lacks a caller until its intended
future role is either documented as future-only or rejected. Future design
belongs in `PLAN.md`, not in silently active production code.

## Phase 8: Make tests justify themselves

For every test, state what current invariant it protects. Remove tests that:

- preserve deleted compatibility APIs;
- exercise a deleted parallel implementation;
- only assert implementation details with no architectural value;
- duplicate stronger integration coverage.

Retain focused tests for population bounds, event semantics, arbitration,
silence/EOS distinction, lifecycle isolation, input boundaries, recurrent
state, session delta merging, and event reward accounting.

Add a small supported-API smoke test so future cleanup cannot accidentally
reintroduce a second path.

## Phase 9: Documentation and repository hygiene

Update `ARCHITECTURE.md`, `PLAN.md`, `README.md`, exports, and examples to
describe only supported code. Mark genuinely future ideas as future ideas.

Remove generated artifacts such as `__pycache__`, `.pytest_cache`, and stale
egg-info from the repository if tracked or present in the working tree, and
ensure ignore rules prevent their return.

## Completion criteria

The pass is complete when:

- no production API accepts raw population offsets;
- no compatibility aliases/wrappers remain without an explicit written reason;
- one canonical language path and one canonical event-output path exist;
- every config option is consumed and justified;
- every retained test protects a current invariant or active experiment;
- docs match the code and clearly separate implemented/deferred work;
- the full test suite passes after deletion, not merely after preserving old
  tests through shims.
