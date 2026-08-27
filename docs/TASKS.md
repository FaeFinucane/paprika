# Tasks

This file contains deferred or actionable work. Implemented phases and historical
cleanup decisions are intentionally not listed as active tasks.

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

- Completed: the shared runtime has seeded, vectorized low-rate background drive;
  normal output projections remain present and only explicit ablations zero edges.
  Runtime and synthetic results expose hidden/output activity, event rate, and
  eligibility/weight changes by pathway.
- Completed: synthetic STDP uses actual immediate and causal delayed outputs,
  preserves eligibility until the scalar event reward, and has an explicit early
  versus reliable asymmetric reward schedule. Penalties remain event-level and
  are not output-volume scaled.
- Completed: `SpikingRuntime.population_diagnostics` reports per-population rate
  mean/spread, active/silent/saturated fractions, voltage and threshold
  distributions, and output event rate. `pathway_diagnostics` reports weight
  and eligibility norms for direct, hidden, and recurrent output pathways.
- Completed: optional slow population homeostasis adds one bounded shared current
  per configured population. It targets a population mean rate, is disabled by
  default, and does not adapt individual neurons, thresholds, output feedback,
  gates, timer neurons, or STDP weights.
- Future research only: investigate whether heterogeneous intrinsic oscillatory
  or pacemaker-like activity is useful for temporal coordination. Do not add
  timer neurons or timer-driven activity to the current baseline experiments;
  begin with stochastic background drive instead.
- Measure recurrent-state persistence and capacity, and decide whether a
  dedicated learned context population is justified. Do not infer this from
  longer examples alone.
- Compare stateful-lifetime evaluation (continuing membrane, refractory,
  pending-current, and recurrent state) with explicit cold-start evaluation;
  report training and evaluation metric windows separately.
- Use the existing sparse weight-delta/session snapshots to distinguish
  short-term neural-state learning from synaptic consolidation, including
  pathway-specific eligibility and weight changes.
- Add stronger dedicated event-stream tests for `ma`, `ba`, `mama`, `baba`,
  repeated characters, valid silence, premature EOS, missing EOS, and post-EOS
  suppression.

- Retain `experiments/run_conversation.py` only as an explicitly named typed
  action baseline. If it is not an active comparison, remove it too rather
  than keeping an undocumented second experiment path.

## Scaling

- Increase network size only after event-stream behavior and activity/capacity
  measurements demonstrate a genuine bottleneck.

## Codebase structure and quality

Completed: `conversation_agent.py` is now a thin public facade; configuration,
response lifecycle/output, and event training live in focused modules. Runtime
network/session state has one owner, and failed raw-event streams clean up their
transient state. Ruff and mypy are configured in `pyproject.toml`; pytest,
Ruff, mypy, and the synthetic experiment are the documented verification set.

Architectural requirements are enforced by the production APIs and tests:
population layout owns addressing, and `EventReadout` owns event output.
Formatting and import/lint rules are style gates supplied by Ruff.

## Safety and resources

- Keep resource budgeting separate from affect. Wire `AffectiveState.energy`
  into computation limits only when an explicit budget policy exists.
- Design a separate normative/safety layer with explicit constraints and
  transparent objectives; do not add a single morality scalar or claim a
  normative veto before it is implemented.

See [architecture](ARCHITECTURE.md) and [training](TRAINING.md) for the current
implemented boundary.

Task adapters own encoding, session boundaries, readout, and reward semantics.
They compose the shared runtime and must not construct or step a second network.
New external stimulation should implement the `Drive` protocol; diagnostics,
homeostasis, and learning observations should implement a scheduled
`NetworkPlugin`.
