# Conversation Architecture

This document describes multi-turn interaction above the semantic Channels
defined in [`INTERACTION_ARCHITECTURE.md`](INTERACTION_ARCHITECTURE.md).

A Conversation gives meaning to a sequence of input and response exchanges.
It does not own the SNN, advance its clock, or directly manipulate neural
state. A running `Session` executes the conversation.

## Conversation and session

The distinction is:

```text
Conversation = semantic history and turn protocol
Session      = live execution context and owned mutable state
```

A single session may contain a conversation with many turns. By default, the
network can continue from one turn into the next, preserving whatever neural
state the session policy allows to persist.

## Turns

A conversation is an ordered sequence of turns. Each turn contains:

1. an input boundary;
2. zero or more input feature activations;
3. zero or more output events, which may occur while input is still being
   received or after input ends; and
4. a completion condition, such as EOS, timeout, cancellation, or session
   failure.

```text
input boundary
    → input features and output events may interleave
    → input end (optional before response completion)
    → EOS / timeout / cancellation / failure
    → next turn
```

A turn should record semantic inputs and outputs, timing, completion status,
and relevant errors. It should not record neuron offsets or require callers to
understand population geometry.

## Input boundaries

For language input, the input sequence is explicitly framed by boolean feature
signals:

```text
INPUT_BEGIN → content features → INPUT_END
```

`INPUT_BEGIN` must occur before content features and `INPUT_END` must occur
after them. Each language input has exactly one of each boundary signal; the
content between them may contain zero or more feature activations. These are
ordinary features delivered through an `InputChannel`, not hidden control
operations that bypass the network.

The boundary policy belongs to the Conversation/input protocol rather than the
SNN. Other input domains, such as action or affect streams, may use different
framing or remain continuously active. Conversation must therefore not assume
that every feature population has language-style begin and end features.

## Responsibilities

`Conversation` should:

- maintain ordered turn history;
- enforce valid turn transitions;
- track input and response activity without requiring exclusive phases;
- permit output events during input as well as after input ends;
- accept response events from `OutputChannel`/`EventOutput`;
- recognise EOS and other explicit completion conditions;
- preserve output-free and non-EOS turns as valid outcomes; and
- expose semantic history to evaluation or host code.

It should not:

- advance the SNN clock;
- encode features into current;
- inspect raw neuron arrays;
- apply learning updates; or
- decide session reset and snapshot policy.

## Response completion

An EOS feature is a semantic output event, normally produced by an
`OutputChannel` and interpreted by `EventOutput`. Conversation logic uses that
event to complete the active turn, but EOS is not required for an output to be
valid. A network may emit one or more ordinary output events without ever
emitting EOS. In that case, the active turn completes when its response
timeout expires, or earlier through cancellation or session failure. These
outcomes should be recorded distinctly from a normal EOS response.

The timeout should be explicit and configurable, expressed in SNN ticks or an
equivalent session clock. It should begin at the chosen response boundary—for
example, at turn start for streaming conversations or at input end for delayed
responses—and the Session architecture should supply the scheduling policy.

After a turn completes, the conversation may begin another input boundary
without requiring a new SNN instance. Whether membrane state, pending current,
detector state, or other state is reset between turns belongs to the Session
policy, not the Conversation.

## Boundary with future layers

Evaluation should consume completed turns or conversation history and produce
an outcome. A later reward policy may convert that outcome into a learning
signal. Neither evaluation nor reward should be embedded in Conversation:
conversations must remain usable for inference and interaction without
training enabled.

The future Session architecture will define how a Conversation is executed,
including channel scheduling, SNN ticks, response limits, snapshots, and
failure isolation.
