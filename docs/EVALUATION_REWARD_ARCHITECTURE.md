# Evaluation and Reward Architecture

This layer judges semantic interaction outcomes and, when training is enabled,
turns those outcomes into learning signals. It consumes completed turns or
conversation history from
[`CONVERSATION_ARCHITECTURE.md`](CONVERSATION_ARCHITECTURE.md); it never reads
neuron offsets or raw SNN state.

## Separation of concerns

Evaluation and reward are related but not the same responsibility:

```text
Conversation events
        ↓
     Evaluation
        ↓
   task outcome / score
        ↓
   Reward policy
        ↓
   reward or RPE
        ↓
     Learning
```

Evaluation should also work when learning is disabled. Learning may likewise
consume supervised, intrinsic, or externally supplied signals without an
evaluation report.

## Evaluation

Evaluation should own:

- matching expected and observed feature events;
- event order and timing;
- missing, unwanted, premature, or post-EOS output;
- valid silence and response completion; and
- turn-level and conversation-level reports.

An evaluation report should preserve both detailed accounting and aggregate
metrics. The report is semantic data suitable for inspection, testing, and
reward calculation.

## Reward

A reward policy maps an evaluation outcome to a numeric learning signal. It may
include stages, baselines, latency penalties, or task-specific weighting, but
those policies should not be hidden inside `Conversation` or `EventOutput`.

Reward application must be explicit and idempotent: one accounted outcome must
not accidentally train the same interaction twice.

## Reward and value

The reward returned for an outcome is distinct from reward prediction error:

```text
reward system produces r
critic predicts V(current_state) and V(next_state)
RPE = r + gamma × V(next_state) − V(current_state)
```

Evaluation supplies the outcome; Learning owns the value comparison and weight
modulation. Evaluation should not inspect raw neuron state or decide which
synapses are eligible.

The reward system may combine external evaluation with internal drives. Early
training can use a strong external signal to teach associations in the internal
reward populations. External reward can later become sparse or corrective while
learned internal reward contributes more strongly. Any internal reward source
must be explicit—homeostasis, curiosity, social engagement, task progress, or
another configured drive—because text prediction error alone does not define
valence.

Reward timing is part of the protocol. A completed response may receive one
turn-level reward after a delayed evaluation. Silence while the network is
processing should not automatically be scored as failure. If a reward is
omitted at an expected evaluation boundary, the resulting negative RPE belongs
to Learning rather than to the semantic evaluation report.

## Deferred decisions

- exact evaluation report shape;
- whether rewards are per event, per turn, or per conversation;
- baseline and RPE ownership;
- reward schedules and training stages;
- internal-reward population inputs and readout;
- external/internal reward blending and calibration;
- how partial, timed-out, cancelled, or failed conversations are scored; and
- how reward-accounting identities prevent duplicate training.
