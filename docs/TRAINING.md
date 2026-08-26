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

Each presented frame advances `SpikingNetwork`. `RewardModulatedSTDP.observe()`
maintains pre/post traces and edge eligibility from spikes. It does not change
weights until a later reinforcement call.

Implemented in `src/continual_agent/simulation/network.py` and
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

`EventReadout` uses hysteresis: a new output must reach
`activation_threshold`, then remains globally active until its population falls
below `release_threshold`. This prevents sustained activity from repeating an
event while allowing the same output to fire again after release.

### 4. Ordered character teacher alignment

`train_response_events(act, target_events)` presents the action context once,
then teacher-presents each target character or `<EOS>` on the existing
`OUTPUT_CHAR` population. `SpikingCharacterDecoder.align_next_token()` updates
the context-to-first-token projection, while `align_recurrent_token()` updates
existing recurrent edges into later target populations. Decoded characters are
not fed back through external `INPUT`, and no new language-state or feedback
population is created.

Implemented in `src/continual_agent/agent/conversation_agent.py` and
`src/continual_agent/language/spiking_decoder.py`.

### 5. Event-stream evaluation

`evaluate_event_stream()` compares target and observed ordered events with
tolerant timing. It accounts for correct and incorrect events, missing output,
valid silence, unwanted output, premature EOS, missing EOS, and post-EOS output.
The resulting `EventStreamReport` retains per-event rewards and metrics.

Implemented in `src/continual_agent/evaluation/event_stream.py`.

### 6. Reward commitment

`train_response_stream()` generates a response, evaluates its event stream, and
passes the report to `apply_event_stream_reward()`. The agent retains records in
`reward_ledger`, computes prediction error against `reward_baseline`, applies
the aggregate error to eligible synapses through reward-modulated STDP, and
resets traces after commitment.

Implemented in `src/continual_agent/agent/conversation_agent.py` and
`src/continual_agent/plasticity/stdp.py`.

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

Implemented by `SpikingRuntime.clone_state()` and
`SpikingRuntime.execute_isolated()` in `src/continual_agent/agent/spiking_runtime.py`;
`ConversationAgent` supplies only task-state transfer hooks.

Every runtime tick also updates windowed diagnostics for each named population.
Rates are spikes per neuron per tick; active and silent fractions count neurons
with at least one or zero spikes in the window, and saturated counts rates at or
above 0.5 by default. Voltage and threshold summaries are distributions, while
weight and eligibility norms are available by output pathway.

Population homeostasis is opt-in through `homeostasis_enabled` and its target,
strength, interval, bound, and population settings on `AgentConfig` or
`TemporalExperimentConfig`. It adds a bounded shared current after each slow
window. The default is disabled; no individual firing-rate targets, adaptive
thresholds, output feedback, gate, or timer mechanism is involved.

## Current boundary

Raw-frame training is an explicit session: exactly one `INPUT_BEGIN` and
`INPUT_END` are required, every frame must be inside that window, and cleanup
resets traces, neural state, readout state, and the training lifecycle even when
the iterator or validation fails. The canonical character training operation
is an ordered event stream ending in EOS and remains a recurrent-state baseline,
not a general sequence-memory claim. The automated curriculum entry point is
`src/continual_agent/experiments/run_conversation.py`.
# Training Protocol

Input presentations begin and end with dedicated neural boundary currents.
Supervised character training aligns EOS from activity propagated after the genuine
`INPUT_END` tick and does not fabricate a boundary feature frame. Reward-modulated STDP never injects a teacher output: it
evaluates actual network events and applies its delayed reward afterward.
