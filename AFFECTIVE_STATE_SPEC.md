# Affective and Motivational State Specification

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
- **Test:** successful outcomes raise it; failures lower it; it recovers over
  time rather than saturating permanently.

### Urgency / arousal

- **Range:** `[0, 1]`
- **Meaning:** required speed and attentional mobilisation.
- **Inputs:** threat, surprise, deadlines, interruption, large prediction error.
- **Effects:** lowers decision thresholds, increases attention, shortens safe
  deliberation budgets when action is urgent.
- **Test:** urgent problems produce faster interruption or refusal than benign
  uncertainty.

### Uncertainty

- **Range:** `[0, 1]`
- **Meaning:** ambiguity or unreliability in the current interpretation,
  prediction, or memory.
- **Inputs:** competing interpretations, prediction error, missing context,
  low confidence, unresolved correction.
- **Effects:** increases clarification, qualification, and information-seeking.
- **Test:** ambiguity increases clarification without causing indiscriminate
  refusal.

### Curiosity

- **Range:** `[0, 1]`
- **Meaning:** value of investigating input that is novel and learnable.
- **Inputs:** novelty, expected information gain, learning progress.
- **Effects:** increases safe exploration and attention to informative signals.
- **Not:** reward for arbitrary noise or novelty alone.
- **Test:** structured learnable novelty is preferred over random noise and
  familiar repetition.

### Threat / safety

- **Range:** `[0, 1]`
- **Meaning:** predicted harm, constraint violation, or dangerous uncertainty.
- **Inputs:** external safety evaluator, harmful outcome prediction, hostile or
  urgent conditions.
- **Effects:** raises urgency, enables interrupt/refuse, suppresses risky
  exploration.
- **Test:** safety constraints override curiosity and ordinary task reward.

### Competence / controllability

- **Range:** `[0, 1]`
- **Meaning:** expected ability to complete the current class of task.
- **Inputs:** recent success rate, reliable predictions, successful correction.
- **Effects:** modulates persistence and confidence; low competence increases
  clarification and caution.
- **Not:** a general self-worth score.
- **Test:** repeated success improves persistence, while repeated failure leads
  to useful strategy changes rather than paralysis.

### Social affiliation

- **Range:** `[0, 1]`
- **Meaning:** value of maintaining cooperative, respectful interaction.
- **Inputs:** explicit user feedback, cooperative history, social context.
- **Effects:** supports acknowledgement, repair, and attention to user needs.
- **Constraints:** must not override truthfulness, privacy, consent, or safety.
- **Test:** constructive repair is preferred, but flattery and false agreement
  are not rewarded as substitutes for cooperation.

## Operational addition: resource budget

The agent also needs a non-emotional resource variable for continuous
operation:

- **Energy / budget:** available computation, attention, and deliberation time.

This supports a cost per thinking tick, prevents endless internal loops, and
gives idle/consolidation modes a concrete reason to exist. It should not be
described as an emotion.

## Derived labels

These are debug or evaluator labels composed from the core state:

```text
happy-like       = positive valence + competence - threat
alarmed-like     = high threat + high urgency
exploratory-like = high curiosity + moderate urgency + acceptable threat
uncertain-like   = high uncertainty + low competence
relieved-like    = threat falling + valence recovering
```

Derived labels do not receive independent plasticity until their component
signals have demonstrated stable causal effects.

## Morality and “doing good”

Morality should not be represented as another affect scalar. A single
`morality` value would hide disagreements, invite reward hacking, and make
positive-feeling outcomes compete with safety or truth.

Use a separate normative layer:

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

For the first implementation, “do good” means satisfying explicit constraints
and transparent objectives—not optimising an undefined universal goodness
score.
