# Tasks

This file contains deferred or actionable work. Implemented phases and historical
cleanup decisions are intentionally not listed as active tasks.

## Output protocol

- Add and evaluate a separate `OUTPUT_FEEDBACK` channel only after the event
  protocol has stable coverage. It must not reuse external `INPUT` or become a
  second decision-maker.
- Add an `OUTPUT_GATE` only as a post-selection emission control; it must not
  choose an action or character.

## Learning and stability

- Add configurable plasticity schedules, adaptive thresholds, and other
  homeostatic stabilisation before considering structural rewiring.
- Measure recurrent-state persistence and capacity, and decide whether a
  dedicated learned context population is justified. Do not infer this from
  longer examples alone.
- Add stronger dedicated event-stream tests for `ma`, `ba`, `mama`, `baba`,
  repeated characters, valid silence, premature EOS, missing EOS, and post-EOS
  suppression.

## First viability experiments

- Add a dedicated synthetic temporal-event experiment with a tiny alphabet and
  controlled input frames, rather than English text. Keep the input encoding
  identical between conditions and train fresh agents for each condition.
- Run an **immediate-copy** condition: the network may emit each symbol once it
  becomes available. Evaluate ordered events, repeated symbols, silence, and
  EOS without requiring exact timestamps.
- Run a separate **delayed-copy** condition: present the complete sequence,
  emit `INPUT_END`, then allow the network a generous patience window to emit
  the copied sequence and EOS. This tests recurrent retention after input ends.
- Do not train both conditions into one agent: identical inputs with different
  timing expectations would be ambiguous without an explicit task-mode signal.
- Use untrained, no-learning, shuffled-target, recurrent-ablation, and
  direct-input-to-output-ablation controls. Record event correctness, missing
  and unwanted events, EOS behavior, latency, hidden activity, and weight
  changes. Treat timing as a measured property, not an early hard target.
- Interpret immediate-copy success with delayed-copy failure as evidence that
  basic pathways work but recurrent retention is inadequate. Delayed-copy
  success is evidence for useful internal temporal state, not evidence of
  language understanding.

### Experiment directory cleanup

- Add a dedicated synthetic temporal-event experiment with its own controlled
  input generation, evaluation, and result reporting; it should not resurrect
  the superseded reduced-language curriculum.
- Retain `experiments/run_conversation.py` only as an explicitly named typed
  action baseline. If it is not an active comparison, remove it too rather
  than keeping an undocumented second experiment path.
- The new synthetic experiment should be the only canonical architecture
  viability entry point and should own its input generation, evaluation, and
  result reporting.

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
