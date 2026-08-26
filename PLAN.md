# Network Evolution Plan

This plan describes the next architectural steps for the continuously active
spiking conversational agent. The network is more than a next-character
predictor: `INPUT` is reserved for outside information, while internal state
and explicit feedback carry the network's own context.

## Core behavioural model

The network runs continuously and may emit events whenever appropriate:

- **Silence** means that no output event is emitted during an interval. It is
  not EOS and should not be represented as an EOS-like output.
- **Signals/characters** are discrete output-population events.
- **EOS** is its own discrete output population. Once it emits, the host treats
  the response as complete and ignores later output until the next response.
- Repeated signals are separate events at separate times. They do not require
  routing the previous character back through the external `INPUT` population.

An optional output gate may sit in front of the output populations. When an
output is selected, a separate internal feedback channel can inform the
network of that selected output. This is distinct from external input and
should be treated as an explicit architectural projection.

## Phase 1: Establish population boundaries

Introduce a central, named population layout rather than passing raw offsets:

```text
INPUT, HIDDEN, AFFECT, OUTPUT_ACTION, OUTPUT_CHAR
```

`AFFECT` is a dual-role population: it has an exposed readout, but also
represents internal state that can influence the recurrent network. It is not
merely an output channel. `OUTPUT_ACTION` and `OUTPUT_CHAR` represent distinct
families of network decisions.

The layout should expose safe slices, indices, and semantic subgroups (for
example, a character's output neurons) to the simulator, readouts, decoder,
plasticity, and tests. Configuration should determine population sizes.

During this phase:

- remove duplicated offset arithmetic;
- make projections select source and target populations explicitly;
- fix decoder projection helpers so their `source_start` and `token_start`
  arguments are meaningful;
- preserve existing behaviour and add population-selection tests.

## Phase 2: Define response-session lifecycle and isolation

Define explicit response states and boundaries, for example:

```text
IDLE → RECEIVING_INPUT → RESPONDING → EOS_RECEIVED → COMPLETE
                                      ↘ RESET_FOR_NEXT_RESPONSE
```

Specify what resets and what persists at each boundary: readout state,
neuron voltage/refractory state, synaptic traces, recurrent activity, affect,
working memory, and plasticity eligibility.

Sessions should be able to run independently and in parallel. Each session
starts from a defined neural-state snapshot rather than requiring a previous
session's live activity. The snapshot should include an **initial synaptic
activity state**—for example short-term traces—and need not be completely
inactive.

Each session should operate on its own copy of trainable weights and produce a
validated series of weight updates/deltas. Merge updates explicitly, then
apply them to the shared network using a defined conflict policy; do not merge
whole mutable network objects.

The initial synaptic activity snapshot should itself be updateable over time.
Begin with a simple, measurable aggregation strategy, such as a weighted or
decayed average of session updates, and later compare more robust alternatives.
Keep long-term weights and short-term activity state conceptually separate.

## Phase 3: Make continuous output events explicit

Replace the current fixed token-window assumptions with an event-oriented
readout layer. It should continuously monitor output populations and:

- detect threshold crossings or discrete spike evidence;
- prevent one sustained activation from producing duplicate events;
- permit the same population to emit again after a readout cooldown;
- resolve simultaneous competing outputs deterministically;
- identify EOS independently from silence;
- record event timestamps and inter-event intervals;
- stop the response after EOS.

Do not reset the whole recurrent network between output events. Reset only
response/readout state when beginning a new response, according to explicit
configuration.

## Phase 4: Resolve output-event arbitration

Define deterministic arbitration when output evidence overlaps:

- multiple characters at once;
- character and EOS together;
- action and character together;
- weak versus strong competing evidence;
- repeated events during readout cooldown.

The policy should be observable and tested independently of learning. It should
not be confused with the optional output gate.

## Phase 5: Define input presentation

Use explicit `INPUT_BEGIN` and `INPUT_END` signals. Start with a simple
rate-coded representation: external features are converted to a controlled
spike rate during presentation, with configurable duration and inter-feature
spacing. Rate coding can be slowed during early curriculum stages while
preserving the same semantic input.

Define whether new input is accepted during response, how overlapping messages
are handled, and how input silence differs from the network emitting no output.

## Phase 6: Evaluate internal output feedback

There is currently no output-feedback or gating mechanism. First add and test
an explicit `OUTPUT_FEEDBACK` channel for confirming what was actually emitted
back to the internal network. The host-side readout may initially select an
event from `OUTPUT_CHAR` or `OUTPUT_ACTION`, then inject an encoded confirmation
through this separate internal channel. It must never reuse external `INPUT`.

Keep the output populations as the source of the network's decisions. Feedback
answers “what was emitted?”, rather than becoming a second decision-maker.
Define and test:

- whether feedback represents selected characters, actions, EOS, or all three;
- feedback delay (at least one simulation tick if causally appropriate);
- feedback duration and decay;
- whether feedback is gated, plastic, or fixed initially;
- behaviour when multiple outputs compete.

Start with a simple fixed/controlled pathway so the architecture can be
validated before making feedback synapses plastic.

## Phase 7: Validate recurrent state and memory

Treat the existing recurrent/hidden network as the internal brain rather than
adding a vaguely defined `LANGUAGE_STATE` population. External information
enters through `INPUT`, is transformed by ordinary synaptic and recurrent
activity, and influences future outputs through the network's existing state.
Output feedback, where enabled, enters through its separate
`OUTPUT_FEEDBACK` channel—not through `INPUT`.

```text
outside information → INPUT → recurrent network state
                                  ↙             ↘
                         OUTPUT_FEEDBACK   OUTPUT_ACTION / OUTPUT_CHAR
```

During this phase:

- preserve recurrent network state across output events;
- test whether it retains information needed for later decisions;
- measure the effect of explicit output feedback separately;
- compare recurrent state against the host-side `WorkingMemory` scaffold;
- refactor training to operate over event streams without feeding decoded
  characters back as external input.

Only introduce a dedicated context population if these experiments identify a
clear limitation in the existing recurrent network. Such a population should
have a defined role—such as slower dynamics, persistent activity, restricted
projections, or distinct plasticity—not merely a new name for hidden neurons.

## Phase 10: Explore optional output gating

There is currently no `OUTPUT_GATE`. Consider it only after output event
semantics and feedback are working. A future gate should control whether a
candidate event from `OUTPUT_CHAR` or `OUTPUT_ACTION` is externally emitted;
it should not decide which character or action was selected. The decision
remains in the output populations, while the gate answers only whether that
candidate may be emitted.

Initially evaluate a neural gate against a host-side gating baseline. If the
gate is neural, define its population, inhibition/gain mechanism, timing,
training signal, and observability explicitly. Do not add it merely to create
a second output decoder.

## Phase 8: Define event-stream training and reward

Represent targets primarily as ordered events, for example:

```text
[(t1, "m"), (t2, "a"), (t3, "m"), (t4, "a"), (t5, EOS)]
```

Training should initially be tolerant of timing. The network should be allowed
to take longer to respond, and early curriculum stages should avoid demanding
fixed inter-event intervals or response deadlines. As training progresses,
timing constraints may be introduced gradually as a configurable communication
skill—analogous to moving from slow, patient teaching toward faster exchange.
Timing should remain a secondary objective until output correctness and memory
are reliable.

Training and metrics must distinguish:

- correct output event;
- incorrect output event;
- missing output where one was expected (after a generous or curriculum-based
  patience window);
- valid silence between events;
- unwanted output during an allowed-silence interval;
- premature EOS;
- missing EOS or output after EOS.

Silence must not receive a blanket positive reward. Its reward depends on
whether an event was expected, while the tolerance for waiting is configured
by the current curriculum stage. Slow early responses should not be punished
simply for being slow.

Add metrics for event precision/recall, response latency, optional timing
error, EOS accuracy, premature EOS, post-EOS output, and silence duration.

## Phase 9: Tune plasticity and stabilisation

Add configuration for a training schedule, such as:

```text
plasticity_start
plasticity_end
plasticity_schedule: linear | exponential | cosine
eligibility_decay
rewiring_rate
structural_plasticity_enabled
```

Use stronger exploratory plasticity early, then reduce it for association and
consolidation. Disable or minimise plasticity during evaluation.

Before adding structural synapse formation, add stabilisation such as firing
rate targets, adaptive thresholds, weight normalisation, or another explicit
homeostatic mechanism. STDP strength alone cannot create new synapses.

## Phase 11: Validate capacity before scaling

The current network is approximately 188 neurons:

```text
input 48, hidden 48, affect 28, actions 28, tokens 36
```

Do not increase size until the population refactor, event readout, feedback,
context, and reward semantics are tested. Establish exact event-stream tests
for `ma`, `ba`, `mama`, `baba`, repeated characters, valid silence, premature
EOS, and post-EOS suppression.

Scale only when measurements show a capacity bottleneck: persistent training
and validation plateaus, saturated or inactive populations, task interference,
longer-memory failure, or collapsed representations. A sensible first increase
would be hidden `48 -> 96`, while keeping output populations small and
connectivity sparse. A separate context population should only be added if
Phase 4 provides evidence for one.

## Verification principles

Every phase should add focused tests and retain the existing simulator,
plasticity, affect, working-memory, decoder, and vertical-slice coverage.
Evaluate neural behaviour separately from host-side policy and scoring
overrides so scaffolding is not mistaken for learned network capability.

## Later work

The following should remain deferred until the lifecycle, event protocol,
input presentation, and recurrent-state baselines are stable:

- `OUTPUT_FEEDBACK` and its session-merge implications;
- neural `OUTPUT_GATE`;
- plasticity schedules, structural rewiring, and homeostasis;
- network scaling.

These are valuable experiments, but implementing them before the baseline is
observable would make failures difficult to attribute.
