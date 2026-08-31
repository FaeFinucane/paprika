# Network Construction Architecture

## Synaptic Neural Network

The outcome of this architecture is one clean class owning and running a synaptic neural network. It contains:

- neuron & synaptic architecture/shapes
- synaptic weights
- neuron state: current voltages, spiking, which are in refractory periods
- tick progression
- (later on) snapshotting and loading from snapshot

It does not know how populations, connections, or seeds were specified.

The main observable from a synaptic neural network, other than the current state, are the spikes, from which most of the surrounding agent architecture is derived.

## Boundary with the interaction layer

This document defines the core simulation layer. It owns neuron geometry,
compiled connectivity, mutable neural state, tick progression, and raw spike
frames. It does not own semantic input or output protocols.

The next layer should be documented in `INTERACTION_ARCHITECTURE.md`. That
layer owns feature-population meaning, `InputChannel` and `OutputChannel`,
encoders and detectors, event-oriented I/O, temporal interpretation, and
host-facing protocols. A channel may address individual features within a
population, but the core SNN remains unaware of those semantic operations.

## Construction

The network is built from immutable specifications and compiled outputs:

```text
Sequence[PopulationSpec] → PopulationLayout
Sequence[ConnectionSpec] + PopulationLayout + seed → Connectivity
PopulationLayout + Connectivity → SNN
```

The dependency direction is one-way. Layout does not depend on connectivity.

Validate user-provided specifications at their construction or compilation
boundaries. Internally generated objects should satisfy their invariants by
construction and do not need redundant defensive validation.

## Populations and layout

Inputs, hidden state, outputs, and affects are all populations, but not all
populations have semantic features. Represent that distinction as a tagged
union rather than a specification with several mutually exclusive nullable
fields:

```python
@dataclass(frozen=True)
class NeuronPopulationSpec:
    name: str
    neuron_count: int
    kind: Literal["neurons"] = "neurons"


@dataclass(frozen=True)
class FeaturePopulationSpec:
    name: str
    features: tuple[str, ...]
    feature_width: int
    kind: Literal["features"] = "features"


PopulationSpec = NeuronPopulationSpec | FeaturePopulationSpec
```

### Training-related populations

The initial training concept can be represented inside this one SNN using
distinct populations with different functional roles:

```text
shared recurrent state and memory
        ├──→ actor/output populations
        ├──→ critic/value populations
        └──→ reward-system populations
```

These are not separate network objects. They may share state and projections,
but should have identifiable population handles so Learning can observe their
activity and diagnostics can report their population-coded values. The critic
represents expected future reward, while the reward system produces current
reward from external evaluation or internal drives. A separate modulatory
population or service can represent the signed reward prediction error (RPE).

Values and RPEs may use population codes rather than single scalar neurons;
for example, opponent positive and negative populations can be decoded as a
signed quantity. The core SNN only simulates the spikes and connections. The
meaning of these populations and the calculation of RPE belong to Learning and
Evaluation.

A featureless population, such as hidden state, uses the neuron variant:

```python
NeuronPopulationSpec(name="hidden", neuron_count=48)
```

A feature-bearing population uses the feature variant and derives its count
from uniformly sized features:

```python
FeaturePopulationSpec(
    name="input",
    features=("INPUT_BEGIN", "INPUT_END", "A", "B"),
    feature_width=1,
)
```

The two variants are exclusive. Different feature populations can use
different feature widths. If one population needs heterogeneous feature sizes,
split it into multiple feature populations instead of complicating the core
representation.
Features remain alongside population geometry for now; they are compiled
semantic names within a population, not connection endpoints. Their bounds
are part of the layout because the core and its consumers need a canonical
mapping to neurons. Their meaning, encoding, detection, and event semantics
belong to the interaction layer.

```python
@dataclass(frozen=True)
class PopulationLayout:
    populations: tuple[Population, ...]
    total_count: int
    fingerprint: str

    @staticmethod
    def build(specs: Sequence[PopulationSpec]) -> "PopulationLayout": ...
```

`PopulationLayout` owns all compiled neuron geometry:

- population order and bounds
- feature names and bounds
- feature widths
- total neuron count

No later component calculates population offsets or counts independently.

PopulationLayout exposes individual population handles which are combined with network outputs to extract data about that specific population:

```python
@dataclass(frozen=True, slots=True)
class NeuronPopulation:
    name: str
    bounds: slice

    @property
    def count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class FeaturePopulation:
    name: str
    bounds: slice
    features: Mapping[str, slice]

    @property
    def count(self) -> int: ...

    def feature_bounds(self, feature: str) -> slice: ...


Population = NeuronPopulation | FeaturePopulation
```

`PopulationLayout` is the single owner of compiled population geometry. The
compiled population is a tagged union: only `FeaturePopulation` has named
feature bounds, and only that variant exposes `feature_bounds()`. There is no
separate interaction-layer feature-population wrapper.

## Connectivity

Connections operate over whole populations:

```python
@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    topology: BernoulliTopologySpec
    weight: BimodalWeightSpec
```

Connection names are derived canonically from their endpoints, such as
`input_to_hidden`. Duplicate source/target pairs should be rejected rather
than assigned arbitrary names.

### Topology

The initial topology is Bernoulli, fan-out-like connectivity. Keep it as one
concrete specification until a second topology is needed:

```python
@dataclass(frozen=True)
class BernoulliTopologySpec:
    expected_fan_out: float

    def build_edges(
        self,
        source: Population,
        target: Population,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]: ...
```

The public parameter is the expected average fan-out, not a raw probability.
Connectivity construction converts it to a Bernoulli probability based on the
available target neurons. Each source/target pair is considered independently.
If the requested fan-out exceeds the available target population, compilation
should fail. Self-edges are excluded automatically for recurrent connections.

Topology returns edge arrays rather than constructing weighted sparse matrices;
`Connectivity` owns the final `SparseSynapses`.

### Weights

```python
@dataclass(frozen=True)
class BimodalWeightSpec:
    positive_mean: float
    negative_mean: float
    negative_fraction: float
    positive_spread: float = 0.0
    negative_spread: float = 0.0
    minimum: float = -1.0
    maximum: float = 1.0

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray: ...
```

The initial distribution is bimodal: most edges are sampled around a strongly
positive mean, while a minority are sampled around a slightly negative mean.
Polarity is sampled independently for each edge.

### Compiled connectivity

```python
@dataclass(frozen=True)
class Connectivity:
    layout_fingerprint: str
    synapses: SparseSynapses
    edges: Mapping[str, np.ndarray]
    seed: int

    @staticmethod
    def build(
        layout: PopulationLayout,
        specs: Sequence[ConnectionSpec],
        seed: int,
    ) -> "Connectivity": ...
```

`Connectivity.build()` resolves population names, asks each topology for edge
pairs, asks each weight specification for weights, and records named edge sets
for ablation and diagnostics.

Random streams should be stable per connection, for example:

```text
connectivity/input_to_hidden/topology
connectivity/input_to_hidden/weights
```

The same layout, specifications, and seed must produce the same graph.

## Neuron dynamics

The initial neuron model is leaky integrate-and-fire (LIF). `LIFNeurons` owns
the per-neuron dynamic state:

- membrane voltage;
- emitted-spike state for the current tick; and
- a refractory counter or equivalent refractory state.

On each tick, the neuron model combines external and pending synaptic current,
applies leak and integration, and checks the resulting voltage against its
threshold. A threshold crossing emits a spike, resets the membrane according
to the configured reset rule, and starts the refractory period. Refractory
neurons cannot emit another spike until that period expires; the exact reset,
current handling, threshold, leak, and refractory parameters belong to the LIF
configuration.

Spikes emitted on tick `t` are transmitted to synaptic targets for tick
`t + 1`. The network therefore has no same-tick recurrent feedback, and each
call to `step()` performs exactly one state transition.

## SNN

The compiled layout and connectivity are combined into one small simulation
class:

```python
@dataclass
class SNN:
    layout: PopulationLayout
    neurons: LIFNeurons
    synapses: SparseSynapses
    pending_current: NDArray[np.float64]
    tick: int = 0

    @staticmethod
    def build(
        layout: PopulationLayout,
        connectivity: Connectivity,
    ) -> "SNN": ...

    def step(
        self,
        external_current: Mapping[Population, NDArray[np.float64]] | None = None,
    ) -> Spikes: ...

    def apply_weight_delta(self, delta: NDArray[np.float64]) -> None: ...
```

`SNN.build()` retains the compiled layout, creates neuron state from
`layout.total_count`, and takes already compiled synapses from `Connectivity`.
The optional input map is keyed by compiled population handles. Each value is
a population-local current vector with shape `(population.count,)`; omitted
populations receive no external current. `step()` validates the handles and
vector shapes, places the local drives into the full internal current vector,
combines external and pending synaptic current, advances the neurons,
transmits emitted spikes for the next tick, advances `tick`, and returns
`Spikes`.

The input map contains at most one already-composed drive per population.
Composition of Channel, background, and homeostatic contributors occurs before
the call; overlapping contributors are summed and clamped there.

Compiled population handles must be immutable and usable as mapping keys.
Their identity should include the layout fingerprint and population identity;
feature-bound metadata must not make otherwise equivalent handles unhashable.

### Weight updates

The SNN owns its synaptic weights and is the only component that mutates them.
Learning and other higher layers may calculate an edge-aligned weight delta,
but they must submit it through `apply_weight_delta()` rather than editing the
synapse storage directly:

```python
delta.shape == snn.synapses.weight.shape
snn.apply_weight_delta(delta)
```

`apply_weight_delta()` validates the delta shape, applies configured weight
limits, and preserves active pathway or ablation constraints. It may only be
called between `step()` transitions. Session coordinates that boundary so a
weight update cannot occur halfway through a neural tick.

The class owns mutable simulation state, but not population definitions,
connection specifications, topology, weight distributions, or seeding policy.

### Spikes

Its primary observable output is a layout-free spike frame:

```python
@dataclass(frozen=True, slots=True)
class Spikes:
    values: NDArray[np.bool_]
    tick: int

    def population(self, population: Population) -> NDArray[np.bool_]: ...
```

```python
hidden_spikes = spikes.population(layout.population("hidden"))
```

`Spikes` does not contain a layout. The `Population` handle carries the
semantic identity and compiled bounds, while the observation remains
layout-independent. `tick` identifies the SNN tick that produced the frame;
the SNN is the sole owner of advancing that clock.
