# Interaction Architecture

This document describes the layer above the core synaptic neural network. The
core architecture is documented in
[`NETWORK_ARCHITECTURE.md`](NETWORK_ARCHITECTURE.md).

The interaction layer gives semantic data a way into and out of the network
without making the SNN understand language, events, labels, or host protocols.

## Boundary

The core SNN owns:

- neuron and synapse state;
- compiled population geometry and connectivity;
- tick progression; and
- raw spike frames.

The interaction layer owns:

- feature-population meaning;
- semantic input and output values;
- `InputChannel` and `OutputChannel`;
- population encoders and detectors;
- event-oriented I/O; and
- temporal interpretation, arbitration, and host-facing protocols.

The dependency direction is:

```text
semantic values → channels → encoded neural drive → SNN
SNN spikes → channels → detected feature evidence → semantic events
```

Channels and their codecs must not mutate neuron state directly. They produce
population-local drive for `SNN.step()` and interpret the `Spikes` returned by
it.

## Features

The core `PopulationLayout` compiles named features to canonical neuron
slices. The core `FeaturePopulation` variant is the only owner of that
feature geometry. The interaction layer gives those slices semantic meaning
by querying the population with a string:

```python
character_bounds = output_chars.feature_bounds("a")
```

Features are not connection endpoints. Connectivity still operates over whole
populations. A feature name identifies an addressable semantic slice within a
population, and that slice may be greater than one neuron so that population
codes and redundant representations remain possible.

For example, `OUTPUT_CHARS` can contain one feature for every character while
remaining one population and one output channel. We do not create a channel
per character.

The population handle should retain, directly or indirectly, the layout
identity needed to prevent it being used against a network compiled from
another layout. Such a mismatch must fail rather than silently indexing the
wrong neurons.

## Channels

A channel is a population-level interaction boundary. It accepts a core
`FeaturePopulation` handle and can address individual features by name, but it
is not created separately for every feature. Constructing a channel for a
plain `NeuronPopulation` is invalid because it has no feature namespace.

### InputChannel

An input channel accepts feature-addressed semantic writes and uses a
`PopulationEncoder` to turn them into external neural drive.

```python
@dataclass
class InputChannel:
    population: FeaturePopulation
    encoder: PopulationEncoder

    def encode(
        self,
        features: Sequence[str],
    ) -> NDArray[np.float64]: ...
```

The result has shape `(population.count,)`, not the size of the whole network.
For now, each feature name means that the feature is active for the current
tick. The channel places encoded activity into each selected feature’s local
slice; all other neurons in its population are zero. An encoder may activate
all neurons in a feature slice, distribute activity across the slice, or
produce a temporal pulse. These are interaction policies, not SNN
responsibilities.

### OutputChannel

An output channel accepts raw spikes and uses a `PopulationDetector` to turn
activity into feature-level evidence.

```python
@dataclass(frozen=True)
class FeatureObservation:
    feature: str
    evidence: float
    active: bool
    timestamp: int


@dataclass
class OutputChannel:
    population: FeaturePopulation
    detector: PopulationDetector

    def observe(
        self,
        spikes: Spikes,
    ) -> tuple[FeatureObservation, ...]: ...
```

The detector evaluates the features in the population, rather than requiring
the caller to inspect neuron slices. It may count spikes, calculate rates,
apply thresholds, detect temporal coincidence, suppress duplicates, or return
the strongest feature. Observation timestamps come from `spikes.tick`; callers
do not provide a second clock.

## Encoders and detectors

`PopulationEncoder` and `PopulationDetector` are concrete classes for now.
They should contain the simplest useful encoding and detection logic directly;
there is no need for abstract base classes while there is only one
implementation of each. If a second genuinely different policy appears,
extract the smallest useful interface at that point. Do not build a general
framework of nested pipelines in advance.

```python
class PopulationEncoder:
    def encode(
        self,
        features: Sequence[str],
        population: FeaturePopulation,
    ) -> NDArray[np.float64]: ...


class PopulationDetector:
    def observe(
        self,
        spikes: Spikes,
        population: FeaturePopulation,
    ) -> tuple[FeatureObservation, ...]: ...
```

Encoders and detectors may own temporal state when their representation needs
it. That state belongs to the interaction/session layer and must be reset or
snapshotted according to the surrounding session policy; it does not belong
in neuron state.

## Event input and output

`EventInput` and `EventOutput` are convenience protocols built on channels.

`EventInput` schedules a semantic write to a selected feature, usually for a
defined duration or tick interval:

```python
EventInput(
    channel=input_features,
    feature="INPUT_BEGIN",
    duration=1,
)
```

`EventInput` has no value property yet: the feature name identifies the
boolean signal, and its duration controls how many ticks it is active. It asks
the channel to encode the write. It never injects a spike or edits a neuron
voltage directly.

`EventOutput` consumes observations from an output channel and can filter by
feature, arbitrate between simultaneous candidates, latch active events, and
apply protocol rules such as end-of-sequence handling:

```python
EventOutput(
    channel=output_chars,
    features=("a", "b", "c"),
)
```

One `EventOutput` can consume every feature in `OUTPUT_CHARS`; separate event
outputs are only needed when different consumers require different policies.

The resulting event should identify the semantic feature and its timing and
evidence, without exposing neuron offsets to callers.

## Tick-level flow

One interaction tick should follow this shape:

1. Channels receive semantic writes for the tick.
2. Each `InputChannel` encodes its writes into population-local drive.
3. The interaction layer passes a population-to-drive map to the SNN.
4. The core SNN places those drives into its network-sized state, advances once,
   and returns `Spikes`.
5. Each `OutputChannel` detects feature observations from that frame, using
   `spikes.tick` for timing.
6. `EventOutput` and other consumers interpret those observations.

The interaction layer may run several channels around one SNN tick, but it
must not advance the network implicitly or invent additional neural ticks.

## Non-goals

This layer does not own:

- synapse construction or topology;
- neuron dynamics;
- population offsets or independent geometry calculations;
- learning rules; or
- semantic concepts required by only one host application.

Those concerns either belong to the core SNN or to a still higher-level agent,
training, or evaluation layer.
