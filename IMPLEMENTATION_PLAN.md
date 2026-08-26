# Continual Spiking Conversation Agent

## Goal

Build a small text-only agent that learns continuously while interacting with an
automated conversation environment. The first prototype should demonstrate:

- spiking recurrent dynamics;
- sparse, local connectivity;
- STDP/Hebbian plasticity;
- reward-modulated eligibility traces;
- homeostatic stability;
- explicit internal drives;
- persistent but revisable conversational preferences.

This is a research prototype, not initially a general-purpose chatbot. It will
not use tools, internet access, arbitrary code execution, or a transformer.

## Initial technical choices

- **Language:** Python.
- **Numerics:** NumPy first; keep interfaces simple enough to replace with a
  faster backend later.
- **Time model:** discrete simulation ticks with event-like binary spikes.
- **Neuron:** leaky integrate-and-fire, with optional adaptive threshold.
- **Connectivity:** sparse directed graph, represented in a form that permits
  local synapse updates.
- **Learning:** pair-based STDP plus a three-factor reward modulation rule.
- **Runtime optimisation:** no backpropagation-through-time in the first
  prototype; any later offline optimisation must be a separately measured
  hybrid experiment.
- **Output:** a small intent/action vocabulary before free-form text.
- **Environment:** scripted conversation scenarios with deterministic scoring.

## Decisions to define before implementation

The following choices should be explicit in configuration rather than hidden in
experiment code:

- initial neuron, synapse, and population counts;
- simulation tick duration and maximum deliberation duration;
- excitatory/inhibitory ratio and connection sparsity;
- input encoding (initially character or small-word symbols plus turn markers);
- output action populations and winner/threshold rules;
- learning rates and timescales for membrane, STDP, eligibility, and
  homeostatic updates;
- reward-prediction-error calculation and reward clipping;
- initial drive setpoints and temperament parameters;
- checkpoint format, random seeds, and whether state is reset between runs;
- training, validation, and held-out evaluation scenario sets.

The first experiment should use a deliberately small fixed configuration, for
example a few hundred neurons and a handful of action populations. Scaling is a
later performance question, not an early success criterion.

## Minimal observable control signals

Named signals should be few, behaviourally grounded, and exposed through debug
traces. A signal is only justified if it changes behaviour or provides a useful
measurement; otherwise it is decorative neural fanfic.

### Communicative acts

These are discrete populations selected during deliberation:

```text
answer, clarify, acknowledge, uncertain, revise, refuse, wait
```

They remain active while a response is generated and are visible to the
evaluator and optional user-facing diagnostics.

### Control signals

These gate operation rather than language meaning:

```text
commit_response, emit_token, end_response, interrupt
```

`end_response` is the internal equivalent of `EOS`; `interrupt` handles safety,
new input, or a deliberation timeout.

### Continuous affective/drive state

Avoid a large catalogue of named emotions at first. Use observable continuous
channels whose effects can be tested:

```text
valence, arousal, uncertainty, curiosity, safety/threat, social_affiliation
```

These modulate attention, action thresholds, plasticity, and persistence. More
recognisable emotions can emerge as combinations of these variables.

### Operational modes

Later, an explicit mode gate can control pathways such as:

```text
interact, idle, consolidate
```

The consolidate mode can support sleep-like replay without pretending that
`sleep` is a conversational act.

Every signal must have a causal test: inspect it, perturb it, and compare the
resulting behaviour against an ablation.

## Trainable affective trajectories

Affective signals are functional internal state variables, not claims about
subjective experience. Train their *trajectories* around events rather than
setting a permanent emotion label:

```text
successful outcome:
  valence rises, competence rises, arousal settles

urgent problem:
  threat and arousal rise, valence falls, interrupt/clarify becomes more likely

novel but learnable input:
  curiosity and moderate arousal rise, exploration becomes more likely

useful correction:
  uncertainty rises briefly, revise becomes likely, then competence recovers
```

An automated evaluator can provide event-conditioned target ranges or reward
components for these trajectories. The network still has to produce the state
through its drives and neuromodulatory pathways; the evaluator should not
silently overwrite the state during normal operation.

Acceptance tests should check both internal and external outcomes: the desired
state changes in the expected direction, and the corresponding behaviour
changes without reward hacking or permanent saturation.

## Agent/environment protocol

The environment should expose typed events rather than arbitrary Python calls:

```text
UserMessage(text)
Feedback(label, value)
Timeout()
ScenarioComplete()
```

The agent should emit typed actions:

```text
Answer(intent)
Clarify()
Acknowledge()
ExpressUncertainty()
Revise()
Refuse()
Wait()
```

This keeps the simulator, scenario evaluator, and neural implementation
separable. Free-form text generation can be attached to `Answer` later.

## Learning and evaluation discipline

Every experiment should compare at least:

- an untrained network;
- a network with plasticity disabled;
- a network with reward modulation disabled;
- the complete network.

Record seeds, configuration, scenario order, checkpoints, rewards, action
traces, drive states, firing rates, and changed synapses. A held-out scenario
stream must never update weights. Otherwise apparent learning may just be
memorisation of the curriculum.

## Proposed package layout

```text
src/
  agent/
    config.py
    conversation_agent.py
  simulation/
    clock.py
    neurons.py
    synapses.py
    network.py
  plasticity/
    stdp.py
    eligibility.py
    homeostasis.py
  cognition/
    drives.py
    working_memory.py
    episodic_memory.py
    action_selection.py
  encoding/
    text_encoder.py
    output_decoder.py
  environment/
    protocol.py
    scripted_conversation.py
    scenarios.py
  evaluation/
    metrics.py
    reports.py
  experiments/
tests/
configs/
```

## Core data flow

```text
Conversation event
  -> text encoder
  -> sensory spike populations
  -> recurrent spiking network
  -> intent/action populations
  -> scripted environment
  -> reward and feedback
  -> neuromodulator and drive update
  -> local synaptic plasticity
  -> memory/consolidation
```

Every interaction should produce a record containing the input, selected
action, reward components, drive state, spike statistics, and changed weights.
This makes the system inspectable rather than a mysterious soup of numbers.

## Memory model

Memory should use several timescales rather than forcing every experience into
the same synaptic matrix.

### Working memory

Short-lived recurrent activity maintains the current conversation context. It
decays naturally, but recurrent attractors can keep selected state active for
several hundred or thousand simulation ticks.

Examples:

```text
current user request
candidate interpretations
unfinished response
current uncertainty
```

### Episodic memory

Important interactions are written to a persistent append-only store. A memory
record should contain the event, a compact sparse neural code, timestamp,
participants or topic labels, drive state, reward, and salience.

The network decides whether to write using novelty, reward, correction,
surprise, and explicit importance signals. The store is not a substitute for
learning: it is a hippocampus-like fast memory that prevents every experience
from being immediately compressed into weights.

### Associative and semantic memory

Repeated or replayed episodes slowly alter recurrent synapses. This stores
associations, expectations, habits, and broad concepts. Consolidation should
occur at a lower learning rate than episodic storage, with synaptic scaling or
protected weights to reduce catastrophic forgetting.

### Memory retrieval

The current neural state produces a sparse retrieval cue. A simple first
implementation compares that cue with stored sparse codes and re-injects the
best few memories into a recurrent memory population as spikes. Later this can
be replaced with a learned spiking associative memory.

Memory writes and reads are gated actions, not automatic dumping of the entire
database into every response. Retrieval should be logged, and memories should
be allowed to be uncertain or wrong rather than treated as unquestionable
facts.

## Milestones

### 1. Deterministic simulator

Implement LIF neurons, refractory periods, sparse synapses, seeded random
initialisation, and recording of spikes, rates, and membrane potentials.

**Acceptance tests:** isolated neuron spikes correctly; recurrent network is
stable; identical seeds produce identical traces.

### 2. Text event encoding

Encode a small vocabulary of characters, words, and conversation markers into
spike trains. Include explicit `USER_TURN`, `AGENT_TURN`, and `END_MESSAGE`
events. Decode a small set of action populations:

```text
answer, clarify, acknowledge, uncertain, revise, refuse, wait
```

**Acceptance tests:** encodings are deterministic, turn boundaries survive the
round trip, and output actions are observable.

### 3. Local plasticity

Add pre/post traces and bounded pair-based STDP:

```text
pre before post  -> strengthen
post before pre  -> weaken
```

Add weight normalisation, firing-rate targets, adaptive thresholds, and
inhibitory balance.

**Acceptance tests:** correlated spike trains strengthen; anti-correlated
trains weaken; prolonged activity does not saturate or silence the network.

### 4. Reward-modulated learning

Maintain synaptic eligibility traces. Apply reward or punishment after each
scenario step:

```text
delta_weight = learning_rate * reward_prediction_error * eligibility
```

Keep reward components separate: task success, uncertainty, repetition,
coherence, and safety violations.

**Acceptance tests:** the network learns a tiny repeated association from
reward; reversal learning eventually changes its preferred action.

### 5. Drives and action selection

Implement slowly changing internal variables:

```text
curiosity, uncertainty, safety, coherence, social_contact, competence
```

Each drive has a state, target range, decay/recovery behaviour, and influence
on neuromodulation or action selection. Safety boundaries remain external
hard constraints, not rewards.

**Acceptance tests:** drive states change predictably; different initial drive
weights produce different exploration and clarification behaviour.

### 6. Automated conversation laboratory

Implement a finite-state scenario runner with templated paraphrases. Initial
scenarios:

- ambiguity requiring clarification;
- unknown information requiring uncertainty;
- user correction and revision;
- repeated failed response;
- simple preference learning;
- delayed feedback;
- conflicting conversational cues.

The evaluator returns structured rewards and records the full trajectory.

**Acceptance tests:** held-out paraphrases outperform an untrained baseline;
the agent improves across repeated scenario exposure; behaviour remains
within safety constraints.

### 7. Short text generation

Only after intent learning works, add a small character- or word-level decoder
with a constrained vocabulary and phrase templates. Keep the intent/action
layer separate from language surface generation so evaluation remains clear.

**Acceptance tests:** generated responses match the selected intent, preserve
uncertainty when appropriate, and do not claim actions that did not occur.

### 8. Free-form language progression

Generalise from intents to sequence generation in separate stages:

1. Use a byte or character vocabulary and emit one symbol per decision cycle.
2. Hold the chosen communicative act (for example `clarify` or `revise`) in a
   persistent control population while the decoder runs.
3. Add an `EOS` population and a response-length budget. The decoder chooses
   when to finish; it is not forced to consume its own text as a new external
   prompt.
4. Permit an optional local efference-copy connection from token populations
   back into the decoder. This is recurrent state, not transformer-style
   re-encoding, and should be compared experimentally against decoder-only
   persistent activity.
5. Train local next-symbol associations from text replay with target-clamped
   STDP or predictive-error plasticity.
6. Keep reward-modulated learning for turn-level qualities such as usefulness,
   correctness, coherence, and style rather than exact wording.
7. Evaluate responses using rubrics, validators, and pairwise preferences;
   do not require one canonical string when several answers are valid.

The language decoder should remain downstream of intent, uncertainty, drives,
and working memory. A response can therefore be rewarded for satisfying a
semantic contract while its surface wording remains flexible.

**Acceptance tests:** held-out text improves in next-symbol prediction;
multiple valid responses can receive positive evaluation; the decoder stops at
`EOS`, respects its budget, and preserves the selected intent and uncertainty.

### Later: episodic memory as an agent tool

Expose a persistent episodic store through explicit agent actions rather than
automatically injecting memories into cognition:

```text
memory.write(record)
memory.search(query)
memory.read(id)
memory.forget(id)
```

The agent chooses when to store and retrieve. The host still enforces quotas,
access permissions, data retention, and validation. Memory operations produce
ordinary observations and rewards, so the same online learning machinery can
learn when they are useful.

Sleep-like replay can later be implemented as an explicit `memory.replay`
action or background mode, without making retrieval an invisible specialised
pathway.

**Acceptance tests:** the agent learns to save salient corrections, retrieve a
relevant prior interaction, avoid irrelevant retrieval, and respect forgetting
and storage limits.

## Automated training protocol

Training is a continuous event stream, not a separate inference mode:

1. Generate or load one conversation scenario.
2. Present its user message to the encoder.
3. Run the network for a fixed number of ticks.
4. Select an intent/action.
5. Let the scenario advance and return structured feedback.
6. Update drives, reward prediction, eligibility traces, and synapses.
7. Save the event and metrics.
8. Periodically replay selected memories and checkpoint state.

Use curriculum scheduling: easy deterministic scenarios first, then paraphrase,
delay, noise, and conflicting cues. Keep a fixed held-out evaluation stream so
continual learning is measured rather than merely observed anecdotally.

## Evaluation metrics

- task success rate;
- clarification precision;
- uncertainty honesty;
- correction acceptance;
- reversal-learning speed;
- memory retention and false-memory rate;
- firing-rate distribution;
- synaptic weight distribution;
- catastrophic-forgetting score;
- action diversity;
- consistency with initial temperament;
- safety-constraint violations.

Personality should be evaluated as stable behavioural statistics across many
scenarios, not as a label or a single pleasing response.

## Explicit non-goals for the first version

- raw image or audio perception;
- unrestricted internet access;
- autonomous tool use;
- human-level language generation;
- claims of biological realism;
- training directly on unverified user feedback;
- modifying safety or evaluator code from inside the agent.

## Current implementation status

Implemented and verified:

- deterministic LIF simulator and sparse recurrent synapses;
- STDP, eligibility traces, reward modulation, and homeostasis scaffolding;
- typed conversation acts and automated scenarios;
- affective state and affect populations merged into the main network;
- affect-to-action synapses and causal ablation tests;
- persistent working memory and conversation reset;
- configurable spiking character populations and EOS;
- local teacher-guided token alignment;
- reduced-symbol curriculum and decoder tests.

Current limitation: token generation collapses (for example `mama` may become
`ammm`). The next decoder change must add a dedicated recurrent **language-state
population** inside the main network:

```text
communicative act + working context + previous character
  -> language-state population
  -> next-character population
```

Teacher training must run one character timestep at a time, aligning:

```text
start -> m
m     -> a
ma    -> m
mam   -> a
```

Use ordinary character-to-state and state-to-character synapses. Do not add
self-auditory/efference-copy feedback yet; that is a later experiment.

## Session handoff: immediate next task

1. Inspect `conversation_agent.py` and `language/spiking_decoder.py`.
2. Add configurable language-state populations to the main network layout.
3. Add character-to-state and state-to-character synapses.
4. Refactor `train_response_text` to train sequential state transitions.
5. Generate each character from a fresh spike window while retaining state.
6. Add exact-sequence tests for `ma`, `ba`, `mama`, and `baba`, plus EOS tests.
7. Run `.venv/bin/pytest -q` and the reduced-language experiment.

Do not broaden to full English until reduced sequences are reliable.

## Known risks and mitigations

- **Runaway excitation or silence:** bounded weights, inhibitory balance,
  homeostatic targets, and rate monitoring.
- **Reward hacking:** keep safety as an external constraint and test reward
  components independently.
- **Catastrophic forgetting:** use slow learning, weight protection, and later
  replay; measure retention after every curriculum phase.
- **No useful language from random weights:** begin with typed intents and a
  constrained decoder rather than claiming open-ended conversation.
- **Uninterpretable failures:** save spike rasters, population evidence,
  deliberation duration, drive levels, and per-synapse update summaries.
- **Accidental personal-data retention:** do not connect live user data until
  the explicit memory API, deletion path, and storage policy are tested.
- **Unbounded thinking:** require an adaptive decision threshold and a hard
  maximum deliberation budget.

## First deliverable

The first usable slice should be a command-line experiment that runs a small
scripted conversation curriculum, prints chosen intents and rewards, and saves
plots/logs for spikes, firing rates, drives, and weight changes. The network
should visibly learn at least clarification, uncertainty, and revision before
we make it eloquent.
