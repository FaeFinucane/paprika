# Affective and Motivational State Specification

## Status

The bounded state variables and event updates in this document are implemented
in `cognition/affect.py` and `cognition/affect_circuit.py`. Effects described as
design goals are not automatically supported behavior: only the projections
and tests documented in `ARCHITECTURE.md` are current guarantees.

## Purpose

These variables are functional control state, not claims about subjective
experience. They must be observable, trainable, and causally connected to
behaviour. Human-readable emotion names are derived labels, not primitive
neurons.

All state updates are bounded and logged. Unless otherwise noted, a value uses
the range `[0, 1]` and follows:

```text
state_next = clamp(state + rate * (target - state) + event_input, 0, 1)
```

State changes should be gradual except for urgent threat or interruption.

## Core signals

### Valence

- **Range:** `[-1, 1]`
- **Meaning:** recent outcomes relative to expectation.
- **Inputs:** reward-prediction error, task success, correction, harm.
- **Effects:** influences persistence, exploration, and response warmth.
- **Not:** a claim that the agent experiences happiness.
- **Test:** successful outcomes raise it and failures lower it; `advance()`
  returns transient state toward its baseline rather than leaving it saturated.

### Urgency / arousal

- **Range:** `[0, 1]`
- **Meaning:** required speed and attentional mobilisation.
- **Inputs:** threat, surprise, deadlines, interruption, large prediction error.
- **Implemented effect:** represented in the affect circuit and exposed state.
- **Deferred effects:** decision-threshold changes, attention allocation, and
  shortened deliberation budgets are not implemented guarantees.

### Uncertainty

- **Range:** `[0, 1]`
- **Meaning:** ambiguity or unreliability in the current interpretation,
  prediction, or memory.
- **Inputs:** competing interpretations, prediction error, missing context,
  low confidence, unresolved correction.
- **Implemented effect:** represented in the affect circuit and exposed state.
- **Deferred effects:** automatic clarification, qualification, and
  information-seeking are not implemented guarantees.

### Curiosity

- **Range:** `[0, 1]`
- **Meaning:** value of investigating input that is novel and learnable.
- **Inputs:** novelty, expected information gain, learning progress.
- **Implemented effect:** contributes to affect modulation.
- **Deferred effect:** safe exploration and attention allocation are not
  implemented guarantees.
- **Not:** reward for arbitrary noise or novelty alone.
- **Test:** structured learnable novelty is preferred over random noise and
  familiar repetition.

### Threat / safety

- **Range:** `[0, 1]`
- **Meaning:** predicted harm, constraint violation, or dangerous uncertainty.
- **Inputs:** external safety evaluator, harmful outcome prediction, hostile or
  urgent conditions.
- **Implemented effect:** represented in the affect circuit and exposed state.
- **Deferred effects:** safety vetoes, interruption/refusal, and exploration
  suppression belong to the future normative/safety layer.

### Competence / controllability

- **Range:** `[0, 1]`
- **Meaning:** expected ability to complete the current class of task.
- **Inputs:** recent success rate, reliable predictions, successful correction.
- **Implemented effect:** contributes to affect modulation.
- **Deferred effects:** persistence, confidence, clarification, and caution
  changes are not implemented guarantees.
- **Not:** a general self-worth score.
- **Test:** repeated success improves persistence, while repeated failure leads
  to useful strategy changes rather than paralysis.

### Social affiliation

- **Range:** `[0, 1]`
- **Meaning:** value of maintaining cooperative, respectful interaction.
- **Inputs:** explicit user feedback, cooperative history, social context.
- **Implemented effect:** represented in the affect circuit and exposed state.
- **Deferred effects:** acknowledgement, repair, and user-needs attention are
  not implemented guarantees.
- **Constraints:** must not override truthfulness, privacy, consent, or safety.
- **Test:** constructive repair is preferred, but flattery and false agreement
  are not rewarded as substitutes for cooperation.

## Deferred: resource budget

The agent may later need a non-emotional resource variable for continuous
operation. `AffectiveState.energy` currently only decays and is not yet used to
gate computation or enforce a budget:

- **Energy / budget:** available computation, attention, and deliberation time.

Cost per thinking tick, loop prevention, and idle/consolidation modes remain
deferred. Energy should not be described as an emotion.

## Derived labels

These are debug or evaluator labels composed from the core state:

```text
happy-like       = positive valence + competence - threat
alarmed-like     = high threat + high urgency
exploratory-like = high curiosity + moderate urgency + acceptable threat
uncertain-like   = high uncertainty + low competence
```

Derived labels do not receive independent plasticity until their component
signals have demonstrated stable causal effects.

## Deferred: normative and safety layer

Morality should not be represented as another affect scalar. A single
`morality` value would hide disagreements, invite reward hacking, and make
positive-feeling outcomes compete with safety or truth.

When implemented, use a separate normative layer:

1. **Hard constraints:** permissions, safety, privacy, consent, and prohibited
   actions. These are enforced outside the learned network and can veto an
   action.
2. **Normative objectives:** honesty, non-harm, respect for autonomy,
   cooperation, and user-specified goals. These can contribute structured
   reward and can conflict visibly.
3. **Learned social expectations:** patterns acquired from feedback and
   experience, always marked as uncertain and revisable.

The evaluator can emit structured events such as `policy_violation`,
`harm_avoided`, `honest_uncertainty`, or `consent_missing`. These events may
affect threat, uncertainty, valence, and plasticity, but they do not secretly
write a morality state into the network.

The current prototype has no normative veto or morality state. “Do good” is
therefore not a supported runtime capability; future work must use explicit
constraints and transparent objectives rather than an undefined universal
goodness score.
