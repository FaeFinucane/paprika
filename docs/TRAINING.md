# Training

Training is a set of cooperating components around one recurrent spiking
network. It is not a single end-to-end sequence-learning system. The current
implementation trains typed action responses and ordered character events using
local updates plus delayed reward.

## Components

### 1. Input presentation

`TextEncoder.feature_vector()` creates bounded semantic features and
`TextEncoder.present()` emits rate-coded frames with dedicated neural
`INPUT_BEGIN` and `INPUT_END` currents. `AgentConfig.presentation_speed` holds each frame for a chosen
number of simulation ticks; values above one are an early-teaching slowdown.

Implemented in `src/continual_agent/encoding/text_encoder.py` and orchestrated
by `ConversationAgent` in `src/continual_agent/agent/conversation_agent.py`.

### 2. Network simulation and eligibility

Each presented frame advances `NetworkCore`. `RewardModulatedSTDP.observe()`
maintains pre/post traces and edge eligibility from spikes. It does not change
weights until a later reinforcement call. Synaptic weights have one shared
`[-1.0, 1.0]` bound across structural initialization, supervised updates, and
reward-modulated STDP; there are no out-of-range bootstrap weights.

Implemented in `src/continual_agent/simulation/core.py` and
`src/continual_agent/plasticity/stdp.py`; calls are made by
`ConversationAgent.respond()`, `generate_response()`, and training methods.

### 3. Typed action training

`train_response(text, expected)` runs the normal response path, compares the
`EventReadout` action with the expected `Action`, computes a scalar reward
prediction error, and reinforces the selected action population. Affective
modulation is supplied as a third factor. The action result is also returned as
an `AgentAction`.

Implemented in `ConversationAgent.train_response()` and
`RewardModulatedSTDP.reinforce()`; action groups are defined by
`src/continual_agent/cognition/readout.py` and `PopulationLayout`.

`EventReadout` consumes spike frames only. A named subgroup becomes a candidate
when at least one of its neurons spikes on the current tick.
After an event, that subgroup is latched while it emits spikes and releases on
the first silent tick. This prevents sustained spiking from repeating an event
while allowing the same output to fire again after release.

### 4. Ordered character teacher alignment

`train_response_events(act, target_events)` presents the action context once,
then teacher-presents each target character or `<EOS>` on the existing
`OUTPUT_CHAR` population. `SpikingCharacterDecoder.align_next_token()` updates
the context-to-first-token projection, while hidden-to-output alignment updates
the existing readout pathway. Decoded characters are not fed back through
external `INPUT` or output reafference, and no new language-state population is
created.

Implemented in `src/continual_agent/agent/conversation_agent.py` and
`src/continual_agent/language/spiking_decoder.py`.

### 5. Event-stream evaluation

`evaluate_event_stream()` compares target and observed ordered events with
tolerant timing. It accounts for correct and incorrect events, missing output,
valid silence, unwanted output, premature EOS, missing EOS, and post-EOS output.
The resulting `EventStreamReport` retains per-event rewards and metrics.

Implemented in `src/continual_agent/evaluation/event_stream.py`.

The synthetic temporal experiment deliberately keeps event accuracy separate
from reward. Its early exploration schedule gives small positive values to some
incorrect or unwanted events so that a quiet, untrained network can discover
the output protocol before stronger penalties are introduced. This is an
experimental bootstrapping choice, not a claim that incorrect output is good.
It may be removed if it fails to improve discovery, retention, or eventual
event accuracy. Experiments using this schedule should report event
precision/recall and sequence success independently of aggregate reward, and
compare neutral or negative incorrect-event schedules when results matter.

### 6. Reward commitment

`train_response_stream()` generates a response, evaluates its event stream, and
passes the report to `apply_event_stream_reward()`. The agent retains records in
`reward_ledger`, computes prediction error against `reward_baseline`, applies
the aggregate error to eligible synapses through reward-modulated STDP, and
resets traces after commitment.

Implemented in `src/continual_agent/agent/conversation_agent.py` and
`src/continual_agent/plasticity/stdp.py`.

The intended near-term synthetic-training comparison is event-based reward
commitment: commit reward when an observed event occurs, rather than also
reapplying a delayed aggregate reward at the end of the trial. Event-based
commitment keeps the delay between output and reinforcement short. A final
trial settlement may remain useful for missing or deferred outcomes, but it
must not silently duplicate an event reward. The event-versus-trial choice is
an experiment setting and should be reported with results.

The reward baseline is a running estimate of expected reward. A
baseline-normalized advantage (also called an RPE here) is simply:

```text
advantage = observed_reward - expected_reward
```

Positive advantage strengthens eligible synapses and negative advantage weakens
them; subtracting the baseline reduces updates caused by unsurprising rewards.
This is distinct from restricting which synapses are eligible. The current
prototype intentionally applies diffuse reinforcement while hidden population
roles are still unknown. Pathway or activity-derived assembly masks should be
comparison conditions, not assumptions about specific hidden neurons.

### 7. Affective modulation and alignment

`AffectiveState` updates bounded functional state from reward, uncertainty,
correction, novelty, threat, and social feedback. Its `modulation()` value
scales the plasticity update. `AffectiveCircuit.align()` locally aligns input
features with affect targets, while `decode()` exposes affect population
activity for inspection.

Implemented in `src/continual_agent/cognition/affect.py`,
`src/continual_agent/cognition/affect_circuit.py`, and the orchestration in
`ConversationAgent.train_response()`.

### 8. Isolated training

`execute_isolated()` copies runtime neural state, traces, readouts, affect,
working memory, and weights. It returns the callback result and validated sparse
`WeightDelta` records. No state is merged by default; explicit weight merging
uses the selected conflict policy. This supports separate training sessions
without mutating shared runtime state accidentally.

Implemented by `RuntimeSession.execute_isolated()` and its runtime clone in
`src/continual_agent/agent/runtime_session.py`, exposed by
`SpikingRuntime.execute_isolated()`; `ConversationAgent` supplies only task-state
transfer hooks.

Every runtime tick also updates windowed diagnostics for each named population.
Rates are spikes per neuron per tick; active and silent fractions count neurons
with at least one or zero spikes in the window, and saturated counts rates at or
above 0.5 by default. Voltage and threshold summaries are distributions, while
weight and eligibility norms are available by output pathway.

Population homeostasis is opt-in through a `HomeostasisConfig` supplied as
`NetworkConfig.homeostasis`; the conversational adapter passes
`AgentConfig.homeostasis` through to that network configuration. The config
owns the target, strength, interval, current bound, and population settings.
It adds a bounded shared current after each slow window. The default is
disabled; no individual firing-rate targets, adaptive thresholds, output
feedback, gate, or timer mechanism is involved.

## Current boundary

Raw-frame training is an explicit session: exactly one `INPUT_BEGIN` and
`INPUT_END` are required, every frame must be inside that window, and cleanup
resets readout and lifecycle state while neural state and traces follow the
configured `SessionPolicy`, even when the iterator or validation fails. The canonical character training operation
is an ordered event stream ending in EOS and remains a recurrent-state baseline,
not a general sequence-memory claim. The automated curriculum entry point is
`src/continual_agent/experiments/run_conversation.py`.

Synthetic delayed training should replay the complete temporal input stream,
including unlabeled delay frames. Those frames are presented as ordinary
network time; the learner is not given a target event instructing it to emit
silence. Output observation remains open during the configured response
period, so the network may emit events at its own pace.

For experiments, active membrane, refractory, pending-current, recurrent,
affective, working-memory, eligibility, and background-drive RNG state should
be independently configurable for preservation or reset. Active state is part
of the intended continual-agent intelligence, so cold-start evaluation is a
comparison condition rather than the universal definition of correctness.
Snapshots should include all state needed to reproduce a run, including drive
RNG state.

## Runtime ownership

Input presentations begin and end with dedicated neural boundary currents.
`NetworkCore` advances neurons and delayed sparse synapses; drives provide
current and plugins observe the resulting arrays. `RewardModulatedSTDP` owns
eligibility and weight updates. Generic event-stream training commits aggregate
reward after readout evaluation, while the synthetic STDP curriculum commits
event rewards as soon as outputs occur. Supervised training remains an explicit runtime protocol rather than
a second network implementation. The runtime keeps aggregate diagnostics and
reusable spike buffers, not a historical spike sequence.

The synthetic delayed-copy experiment uses the clearly named
`train_delayed_copy_baseline()` production-runtime path. For each labelled
symbol it clips a structural update to existing HIDDEN->HIDDEN edges sourced by
that symbol's hidden feature group, allowing state to persist across the input
boundary while normal hidden-to-output teacher alignment remains active. It
does not create a second network or retain a host-side symbol queue.
