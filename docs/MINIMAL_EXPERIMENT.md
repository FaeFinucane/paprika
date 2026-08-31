# Minimal Training Experiment

This document defines a small, reproducible experiment for testing whether a
single spiking network can learn increasingly structured text responses through
reward-modulated STDP. It deliberately omits internal reward learning,
long-term memory, and broad language modelling until the basic learning loop is
observable.

## Question

Can a network learn a simple response protocol when:

```text
text input → spiking recurrent state → text output
                         ↓
                   critic/value
                         ↓
teacher reward → RPE → reward-modulated STDP
```

The first experiment should test contingency and response structure, not claim
to reproduce biological language development.

## Network populations

Use one SNN with these populations:

```text
INPUT_CHARS  → HIDDEN
HIDDEN       → OUTPUT_CHARS
HIDDEN       → CRITIC
HIDDEN       → REWARD
OUTPUT_CHARS → HIDDEN
```

Suggested initial sizes:

```text
INPUT_CHARS:  one feature slice per input symbol
OUTPUT_CHARS: one feature slice per output symbol
HIDDEN:       64 neurons
CRITIC:       16 neurons
REWARD:       16 neurons
```

The exact character set should be small, for example:

```text
INPUT_CHARS  = {A, B, ?}
OUTPUT_CHARS = {A, B, EOS}
```

Each character feature may contain several neurons so that output and value
decoding use population activity rather than relying on one neuron.

`CRITIC` and `REWARD` are populations in the same SNN, not separate network
objects. `CRITIC` estimates future reward. `REWARD` produces the current
internal reward estimate. For the first experiment, the reward population may
receive the external reward signal directly or remain a fixed primitive signal;
learning the reward system is a later comparison.

## Neural simulation

Use a simple leaky integrate-and-fire neuron model:

```text
v[t + 1] = decay × v[t] + input_current[t] + synaptic_current[t]
```

When `v` reaches threshold:

```text
emit spike
reset v
enter refractory period, if used
```

Choose one fixed simulation timestep and use it consistently for input pulses,
eligibility decay, output timing, and response timeout. The first experiment
does not need conductance dynamics or multiple biological timescales.

Weights should be bounded after every update:

```text
weight = clamp(weight + delta_weight, minimum_weight, maximum_weight)
```

Use seeded initialisation so runs can be reproduced.

## Input and output protocol

Represent one training example as a short turn:

```text
INPUT_BEGIN → input symbols → INPUT_END
             → network processes and generates output
             → output symbols → EOS
```

The input encoder activates the selected character feature for a fixed number
of ticks. The output detector counts spikes in each output feature slice over an
output window and selects the feature with the greatest evidence above a
threshold. It should suppress duplicate detections within one output window.

For the first experiment, limit responses to one or two symbols followed by
`EOS`. A response timeout completes a turn if `EOS` is not emitted. Timeout,
silence, and malformed output must be recorded separately from a valid response.

Output generation is an actor decision. If several output features are active,
use a deterministic winner during evaluation and a stochastic winner during
training so the network can explore alternatives.

## Training task

Begin with a tiny teacher protocol:

```text
input A → desired response A EOS
input B → desired response B EOS
```

The teacher should provide differential turn-level reward, for example:

```text
correct response and EOS       → +1.0
structured but incorrect       → +0.2
malformed, irrelevant, timeout → -0.2
```

Do not reward every output equally after the first smoke test. The teacher’s
response must depend on the relationship between input and output, or there is
no pressure to learn structure or contingency.

Later curriculum stages can expand the task:

```text
single-symbol copying
→ two-symbol copying
→ simple pattern completion
→ basic question/answer pairs
```

Next-token prediction is not required for this experiment. If added, its loss
or prediction error should be recorded as a separate learning signal rather
than being confused with reward prediction error.

## Value and reward readouts

Decode the critic population into a scalar value using a simple population
average or weighted sum:

```text
current_value = decode(CRITIC activity before output)
next_value    = decode(CRITIC activity after teacher response)
```

For the initial task, decode `REWARD` similarly. The reward system can use
opponent populations if signed internal reward is required:

```text
internal_reward = positive_reward_activity
                  − negative_reward_activity
```

The external teacher reward is the only required reward source for the first
version. Initially use:

```text
total_reward = external_reward
```

Do not let the network define its own reward before this externally anchored
experiment is working.

## Eligibility and STDP

Maintain one eligibility value for each synapse. A minimal pair-based rule is:

```text
pre spike before post spike → eligibility += potentiation_amount
post spike before pre spike → eligibility -= depression_amount
```

After each simulation tick, decay every eligibility trace:

```text
eligibility *= trace_decay
```

The trace must last long enough to bridge output generation and teacher
evaluation, but not so long that all earlier turns remain eligible.

At the response evaluation boundary, apply the signed RPE to every eligible
synapse in the unified network:

```text
delta_weight = learning_rate × RPE × eligibility
```

Then clamp weights and clear or strongly decay the traces associated with the
completed turn. The conceptual update is one network-wide operation; the local
eligibility traces provide population and temporal selectivity.

## RPE calculation

At response completion:

```text
current_value = critic.predict_value(current_state)
output = actor.generate_output(current_state)
reward = teacher.evaluate(output)
next_value = critic.predict_value(next_state)

RPE = reward + gamma × next_value − current_value
```

Use `next_value = 0` for a terminal training example. Start with a discount
factor near `1.0` for short turns. Bound or normalise unusually large RPEs so a
single response cannot destabilise the whole network.

The same RPE modulates eligible actor, critic, reward, and shared synapses. This
does not mean they learn the same function: their inputs, outputs, and activity
patterns differ.

## Turn timing

Use event-based reward timing rather than treating every waiting tick as a failed
action:

```text
input ticks
→ output/thinking ticks: collect spikes and eligibility only
→ EOS, timeout, or cancellation
→ evaluate exactly once
→ calculate RPE
→ apply one weight update
```

If a reward is expected at a boundary and is omitted, represent that omission
as the outcome for that boundary. Silence before the boundary is not itself a
negative reward.

## Measurements

Record at least:

```text
response accuracy
EOS/completion rate
response length
turn reward
current value and next value
RPE magnitude and sign
output-population spike counts
critic-population decoded value
weight and eligibility distributions
```

The minimum success criterion is improvement over a seeded-random actor on the
copying task, together with critic predictions that become calibrated over
repeated trials. A falling RPE alone is not success: it can indicate only that
the reward became predictable.

## Development stages after the minimum experiment

Only after the externally rewarded task is stable should the experiment add:

1. a learned internal reward population trained from external outcomes;
2. intrinsic rewards such as useful novelty or information gain;
3. sparse external feedback and internal reward blending;
4. longer responses and delayed outcomes; and
5. separate next-token prediction alongside RPE-modulated learning.

Each stage should be compared with an ablation that removes the new signal.
Otherwise it will be difficult to tell whether improvement came from
reward-modulated plasticity, ordinary prediction learning, or a conveniently
overhelpful evaluator.
