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

## Scaling

- Increase network size only after event-stream behavior and activity/capacity
  measurements demonstrate a genuine bottleneck.

## Safety and resources

- Keep resource budgeting separate from affect. Wire `AffectiveState.energy`
  into computation limits only when an explicit budget policy exists.
- Design a separate normative/safety layer with explicit constraints and
  transparent objectives; do not add a single morality scalar or claim a
  normative veto before it is implemented.

See [architecture](ARCHITECTURE.md) and [training](TRAINING.md) for the current
implemented boundary.
