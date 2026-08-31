# Training Concept

This document describes a biologically inspired training concept for the
remade spiking network. It is a conceptual model rather than a final learning
API. The core idea is one SNN containing distinct functional populations whose
eligible synapses are modulated by a shared reward prediction error (RPE).

## Functional populations

The network may contain the following populations:

```text
input/text populations
          ↓
shared recurrent state and memory
          ├──→ actor/output populations
          ├──→ critic/value populations
          └──→ reward-system populations
```

- **Actor/output** chooses or generates the next response.
- **Critic/value** estimates expected future reward from the current state.
- **Reward system** produces current reward from external evaluation and/or
  internal drives.
- **RPE/modulatory system** compares received reward with predicted value and
  provides a signed plasticity signal.

Actor and critic are functional roles, not necessarily separate networks. They
may share recurrent representations and memory while retaining separate value
and output populations. The reward system is related to the critic but is not
the same thing:

```text
reward system: current reward, r
critic:        predicted future reward, V
```

## Population-coded values

Rewards, values, and RPEs do not need to be stored as single numeric neurons.
They can be represented by population activity and decoded when required:

```text
positive-reward activity − negative-reward activity = signed reward estimate
critic population activity                               = expected value
positive-RPE activity − negative-RPE activity             = RPE
```

Feature populations can also represent text input and output. The interaction
layer owns the semantic mapping between features and population activity; the
core SNN only produces raw spikes.

## Reward-modulated plasticity

Spike timing creates a local, temporary eligibility trace at a synapse:

```text
pre/post spike timing → eligibility trace
```

The trace identifies recent activity that may have contributed to an outcome.
It decays while the network continues processing. When an evaluation arrives,
the RPE converts eligible traces into lasting weight changes:

```text
weight change ∝ local eligibility × signed RPE
```

The RPE may be broadcast to the network, but it should not strengthen every
recently active synapse. Local spike timing, circuit connectivity, population
gating, and trace decay provide approximate credit assignment.

## Value and RPE calculation

The critic learns expected future cumulative reward through temporal-difference
bootstrapping. For a state transition:

```text
current_value = V(current_state)
next_value    = V(next_state)

RPE = reward + gamma × next_value − current_value
```

The reward is the outcome that actually occurred. `gamma` discounts future
reward. Initially, value predictions may be arbitrary; repeated outcomes train
them through the RPE:

```text
prediction too low  + reward arrives       → positive RPE
prediction too high + reward is omitted    → negative RPE
prediction accurate                         → near-zero RPE
```

An omitted reward should only produce a negative RPE at an evaluation point
where reward was expected. Silence during internal processing is not
automatically a negative reward.

## High-level interaction loop

```text
state = current text/history + internal state

current_value = critic.predict_value(state)
output = actor.generate_output(state)

while generating or thinking:
    observe spikes
    update local STDP eligibility traces
    do not treat waiting as a failed evaluation

next_state = teacher/environment response
external_reward = teacher/environment evaluation
internal_reward = reward_system.evaluate(state, output, next_state)
total_reward = external_reward + internal_reward

next_value = critic.predict_value(next_state)
RPE = total_reward + gamma × next_value − current_value

network.apply_reward_modulated_stdp(RPE)
state = next_state
```

In one unified network, the final operation can affect eligible synapses in the
actor, critic, reward system, and shared representations. The same modulation
does not make those populations learn the same function: their inputs,
connectivity, activity, and eligibility traces give them different roles.

Learning updates should become visible at an explicit turn or training-step
boundary, rather than mutating weights halfway through a neural transition.

## Structured output and language learning

RPE does not directly mean “be structured.” Structured output emerges when it
reliably produces better consequences:

```text
any output
    → contingent output
    → turn-taking
    → imitation/copying
    → structured continuation
    → basic question-answering
```

Early social or teacher feedback can reward successive approximations without
explicitly labelling every value. The evaluator should respond more richly to
relevant, contingent, or successful outputs and provide little or no reward for
irrelevant output.

If next-token learning is also used, keep its signal conceptually distinct:

```text
token prediction error → improve language/world modelling
RPE                   → improve value prediction and response selection
```

Surprise is not automatically bad, and reducing prediction error is not the
same as learning valence.

## Development of internal reward

The reward system may begin with primitive, broad signals analogous to
homeostatic and social drives:

```text
internal state restored       → positive reward
internal state disrupted      → negative reward
useful novelty or information  → intrinsic reward
social engagement or approval  → positive reward
```

External reward can train the reward system to associate stimuli, contexts, and
outcomes with internally generated value. A developmental transition can then
move from externally anchored reward to increasingly internal reward:

```text
early:   strong external reward, weak primitive/internal reward
middle:  external reward trains and calibrates internal value
later:   internal reward dominates; external feedback is sparse or corrective
```

Internal reward is emergent from experience, but not causeless. Its eventual
values remain shaped by initial drives, architecture, early feedback, and
interaction history. Removing every evaluative signal would not create values
from nothing; it would remove the basis for reward learning.

## Limitations and safeguards

This is an approximation of biological learning, not a complete brain model.
Eligibility traces solve some temporal credit assignment but do not prove which
synapses caused an outcome. Repeated experience, circuit separation, gating,
and suitable trace timescales are needed to reduce accidental reinforcement.

Large sparse completion rewards may be noisy and slow to learn from. Prefer
meaningful response boundaries, decaying eligibility traces, critic estimates
at multiple timescales, and carefully bounded or shaped rewards. Do not punish
every silent timestep unless silence is deliberately defined as an action with
negative consequences.

The reward signal should also remain distinct from the RPE:

```text
reward system produces r
critic predicts V
RPE compares r with V
RPE modulates eligible plasticity
```
