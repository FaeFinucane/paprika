# Dopamine-population RPE redesign: implementation plan

Status: implementation in progress.

Implemented: compiled transmitter/dopamine-response traits; non-negative
synaptic strengths with structural signs; named projection policies and delays;
unipolar rate inputs/readouts; dopamine-opposed STDP; intrinsic-bias,
inhibitory, and scaling homeostasis; explicit transition-boundary TD oracle;
and a shared reference dopamine-circuit builder. The legacy numeric/RPE
experiments have been removed.

Still an empirical gate: the transition-gated neural previous-value memory is
not promoted over the explicit reference trace until it matches the reference
comparator and demonstrates stable learning across seeds. The code keeps the
reference temporal term isolated for precisely that comparison.

This is a clean-break redesign. It replaces the current numeric-channel and
reward/predictor implementation; it is not a compatibility migration.

## 1. Goal

Build a rate-coded spiking circuit in which:

- external reward and punishment are separate unipolar neural inputs;
- positive and negative predicted value are separate unipolar neural
  assemblies;
- one tonic `DOPAMINE` population represents signed reward-prediction error
  (RPE) as firing above or below its baseline;
- dopamine-aligned and dopamine-opposed neurons learn with opposite responses
  to the same dopamine deviation;
- excitation and inhibition continue to obey Dale's principle; and
- activity remains sparse and recoverable through fast inhibition, explicit
  bounded homeostatic drive, and very-slow synaptic scaling.

The implementation must answer the learning hypothesis, not merely produce a
dopamine-shaped signal. The circuit must implement and test:

```text
delta_t = r_t + gamma * V_t - V_(t-1)
```

A burst or dip that does not shrink or transfer as prediction improves is not
an adequate substitute for TD error.

## 2. Non-negotiable constraints

1. **No backwards compatibility.** Delete deprecated types, fields, adapters,
   diagnostics, and experiment wiring at final migration. During construction,
   the new vertical slice may coexist temporarily, but do not connect the two
   designs with adapters or retain both behind runtime flags.
2. **No `NumericChannel`.** Neural magnitudes are decoded from smoothed
   population firing rates, never positive-spike count minus negative-spike
   count.
3. **All represented magnitudes are unipolar.** Signed reward and signed value
   use opponent populations. The sole signed readout is dopamine's deviation
   from a nonzero tonic rate.
4. **Dale's principle remains structural.** An ordinary neuron does not change
   the sign of its outgoing synapses. Net sign changes use an intervening
   inhibitory neuron or population.
5. **Dopamine response and transmitter sign are independent traits.** A neuron
   may be excitatory or inhibitory and independently dopamine-aligned,
   dopamine-opposed, or dopamine-neutral.
6. **Connection purpose is explicit.** Every projection is fixed,
   dopamine-plastic, or inhibitory-homeostatic. Scaling eligibility is also
   explicit. No learning rule selects synapses merely by the sign of their
   source neuron.
7. **Comparator coefficients are protected.** Reward and TD-comparator paths
   are fixed and excluded from reward learning and synaptic scaling.
8. **Time is defined at state-transition boundaries.** `V_(t-1)` means the
   latched value for the previous evaluated state, not an arbitrary previous
   simulator tick.
9. **Homeostasis must not manufacture RPE.** Homeostasis operates more slowly
   than task signals. The phasic return of `DOPAMINE` comes from cessation of
   comparator drive and fast circuit dynamics, not slow adaptation of its
   target rate.

## 3. Resulting architecture

```text
                             learned, dopamine-aligned
                         +------------------------------+
                         |                              v
task inputs <----> recurrent HIDDEN ------------> POSITIVE_VALUE
                         |                              |
                         | learned, dopamine-opposed    | + gamma (current)
                         +--------------------------> NEGATIVE_VALUE
                                                        | - gamma (current,
                                                        |   via inhibition)
                                                        |
REWARD_POSITIVE -- fixed excitation --------------------+
REWARD_NEGATIVE -- fixed inhibition --------------------+----> DOPAMINE
                                                        |       tonic rate
POSITIVE_VALUE -- previous-value inhibitory path -------+           |
NEGATIVE_VALUE -- previous-value inhibitory path -------+           v
                                                       signed modulation:
                                                   rate - tonic baseline
                                                               |
                                                               v
                                                eligible network synapses
```

The diagram shows net effects on `DOPAMINE`, not permission to violate Dale's
principle. `NEGATIVE_VALUE` is an inhibitory population, so its firing rate can
directly supply the current negative comparator term.

### 3.1 Functional populations

| Population | Ordinary output | Dopamine response | Resting activity | Role |
| --- | --- | --- | --- | --- |
| Task input/output | excitatory | neutral or configured | event-driven | External interaction |
| `HIDDEN` principal neurons | excitatory | aligned, opposed, or neutral | sparse | Shared state, prediction, and action representation |
| `HIDDEN` stabilizers | inhibitory | neutral initially | sparse | Fast feedback inhibition |
| `POSITIVE_VALUE` | excitatory | aligned | low/sparse | Unipolar magnitude of expected positive value |
| `NEGATIVE_VALUE` | inhibitory | opposed | low/sparse | Unipolar magnitude of expected negative value |
| Comparator relays | excitatory or inhibitory by population | neutral | event-driven | Fixed current/previous-value signs and delays |
| `REWARD_POSITIVE` | excitatory | neutral | silent without input | Unipolar appetitive outcome input |
| `REWARD_NEGATIVE` | inhibitory | neutral | silent without input | Unipolar aversive outcome input |
| `DOPAMINE` | modulatory broadcast only | neutral | tonic | Signed RPE encoded around baseline |

`HIDDEN` remains one shared recurrent model in the conceptual architecture.
Its neurons may have different traits; this does not create separate actor and
critic networks. Value and action learning use the same general three-factor
rule over the same recurrent state.

### 3.2 Exact comparator signs

Decode the two unipolar value rates as:

```text
V_t = V_positive_t - V_negative_t
```

Then:

```text
delta_t = (r_positive_t - r_negative_t)
        + gamma * V_positive_t
        - gamma * V_negative_t
        - V_positive_(t-1)
        + V_negative_(t-1)
```

The net pathways into `DOPAMINE` must therefore be:

| Source term | Net effect on dopamine |
| --- | --- |
| Positive reward | excitation |
| Negative reward | inhibition |
| Current positive value | excitation with gain `gamma` |
| Current negative value | inhibition with gain `gamma` |
| Previous positive value | inhibition with gain `1` |
| Previous negative value | excitation with gain `1` |

Initially, a software-held previous-value trace may provide the final two
terms behind a narrow temporal-memory interface. Reward and current-value
terms should still use their fixed neural paths. The trace is a reference
implementation, not the final neural mechanism.

The later neural implementation must preserve transition semantics. A relay
chain with a fixed number of tick delays is only valid if state durations are
fixed. Variable-duration tasks require a transition-gated neural memory or
sample-and-hold circuit for the previous value.

### 3.3 Dopamine readout and plasticity

Let `D` be the smoothed `DOPAMINE` population rate and `D0` its configured
tonic rate. The signed third factor is a calibrated deviation:

```text
dopamine = normalize(D - D0)
```

Use a bounded, piecewise-linear function with a small symmetric deadzone.
Subtract the deadzone before scaling so the function is continuous at the
cutoff. Avoid burst-sensitive EMA rules that change the meaning of an
otherwise identical firing rate.

For a dopamine-plastic synapse into neuron `j`:

```text
delta_strength_ij = learning_rate
                  * eligibility_ij
                  * dopamine
                  * response_sign_j
```

where:

```text
dopamine-aligned: response_sign = +1
dopamine-opposed: response_sign = -1
dopamine-neutral: response_sign = 0
```

This is one learning-rule family. The opposite response is a postsynaptic
neuron trait, not a special negative-value learning algorithm. In particular,
the fact that a D2-like neuron may have inhibitory output does not itself fix
negative learning; its incoming causal synapses still require the opposed
response.

The response trait is stored per neuron and compiled into a per-synapse cache
using the target neuron. A connection may still override learning entirely by
being marked fixed or homeostatic.

## 4. Core software model

### 4.1 Neuron traits

Replace the current implicit conventions with compiled arrays for at least:

```text
ordinary output: excitatory | inhibitory | modulatory
dopamine response: aligned | opposed | neutral
```

Population specifications should declare these traits cleanly. Mixed
functional populations may compile multiple neuron classes into one named
population, but downstream code must consume per-neuron arrays rather than
reconstructing traits from offsets.

Modulatory neurons do not create ordinary signed synaptic current. Reject
ordinary outgoing `DOPAMINE` connections; its firing rate is consumed by the
plasticity system as a broadcast third factor.

Store ordinary synaptic strength as a non-negative magnitude. Derive current
sign from the source neuron's output trait:

```text
synaptic_current = transmitter_sign(source) * strength
```

This keeps Dale's principle in one place and lets the same positive strength
update potentiate either excitation or inhibition. Replace APIs that directly
add signed weight deltas with bounded strength-update APIs.

### 4.2 Connection policy

Extend `ConnectionSpec` and compiled synapses with explicit metadata:

```text
learning: fixed | dopamine_stdp | inhibitory_homeostatic
scalable: bool
```

Required rules:

- reward and comparator projections: `fixed`, `scalable=False`;
- learned excitatory/task/value projections: `dopamine_stdp`, normally
  `scalable=True`;
- stabilizing inhibitory feedback: `inhibitory_homeostatic`, never
  dopamine-modulated;
- unused or diagnostic projections are not silently created.

Weight updates must preserve Dale's principle and connection-specific bounds.
The current network-wide `source is excitatory` learning mask must be removed.

### 4.3 Rate coding

Split input and observation instead of retaining a bidirectional numeric
channel:

- a unipolar rate encoder accepts values in `[0, 1]` and supplies stochastic
  or distributed drive to one population for a defined duration;
- a population-rate observer computes mean firing probability using one fixed
  low-pass time constant;
- a baseline-centred dopamine observer converts rate deviation into a bounded
  signed third factor; and
- rate state has explicit reset and episode-boundary behavior.

Do not add a new numeric population type. Rate coding is a way of driving and
observing ordinary neuron populations.

### 4.4 Activity regulation

Implement three distinct mechanisms:

1. **Fast feedback inhibition:** ordinary excitatory-to-inhibitory and
   inhibitory-to-excitatory circuit paths. These act on neural timescales.
2. **Slow homeostatic drive:** a named, bounded current-source plugin slowly
   changes its output toward a configured population target rate. It must not
   mutate neuron parameters or obscure the source of baseline activity. Do not
   train thresholds with reward.
3. **Very-slow synaptic scaling:** multiplicatively scale only incoming
   excitatory strengths whose connections are marked `scalable`. Apply a common
   target-neuron or population factor so relative learned structure is
   preserved.

Target rates are configured per population and initially hand-tuned. Input
fan-in and population size must be part of initial strength calibration; a
constant total drive is preferable to a constant per-synapse strength.

The architecture has no ambient background-drive mechanism. Any future
stochastic exploration must be an explicitly named, optional influence and
must not be required to rescue a silent network.

## 5. Implementation sequence

Each phase has a completion gate. Do not begin experiment tuning before the
lower-level gate passes.

### Phase 1: make traits and connection ownership explicit

Primary files: `src/network/population.py`, `src/network/connectivity.py`,
`src/network/snn.py`.

1. Add typed neuron output and dopamine-response traits.
2. Compile traits into per-neuron arrays owned by `PopulationLayout`.
3. Add learning/scaling policy to connection specifications and compiled
   synapses.
4. Replace signed stored weights with non-negative strengths; derive current
   sign from source traits.
5. Reject ordinary current-producing connections from modulatory populations.
6. Change strength-update APIs to accept/select explicit connection masks.

Completion gate:

- trait compilation is deterministic;
- Dale's principle is preserved after initialization and every update;
- stored ordinary strengths cannot become negative;
- fixed connections cannot be changed by any plasticity/scaling operation;
- invalid modulatory connections fail at construction.

### Phase 2: replace numeric channels with rate primitives

Primary files: split the numeric code out of `src/interaction/channels.py`
into a focused rate module; update exports and tests.

1. Implement unipolar rate input and population-rate observation separately.
2. Implement the tonic-baseline dopamine readout.
3. Define reset, duration, range checking, smoothing, and normalization.
4. Test rate estimates across population sizes and refractory settings.
5. Keep the new rate primitives independent from the legacy experiments; do
   not create adapters between rate coding and `NumericChannel`.

Completion gate:

- increasing a `[0, 1]` input produces a monotonic decoded rate;
- after input duration ends, drive stops and the observed rate decays toward
  the population's current unforced rate;
- a tonic dopamine population decodes as zero modulation;
- equal bursts and dips give opposite-signed, approximately calibrated
  modulation; and
- the rate primitives contain no positive/negative spike-count convention.

### Phase 3: implement two-sided activity regulation

Primary files: `src/interaction/drives.py` and focused plasticity modules under
`src/interaction/plasticity/`.

1. Keep neuron dynamics shared and fixed; do not add additive neuron bias.
2. Add slow target-rate drive adaptation with bounded, per-population current.
3. Restrict inhibitory plasticity to connections explicitly tagged for it.
4. Add very-slow multiplicative scaling restricted by `scalable` metadata.
5. Construct feedback-inhibitory populations and connections explicitly in a
   minimal stability experiment.
6. Model any tonic or background activity as an explicit, inspectable drive.

Completion gate:

- a silent-but-healthy network returns to sparse baseline without stochastic
  rescue;
- an overactive network returns toward baseline without saturating inhibition;
- removing task input does not collapse all persistent population activity;
- sustained task input is not cancelled on task timescales; and
- fixed reward/comparator strengths remain bit-for-bit unchanged.

### Phase 4: generalize three-factor plasticity

Primary files: `src/interaction/plasticity/hebbian.py` and its tests.

1. Retain one eligibility-trace calculation.
2. Replace the global excitatory-source mask with connection-policy masks.
3. Cache each dopamine-plastic synapse's response sign from its target neuron.
4. Apply the continuous piecewise-linear dopamine deadzone and bounded gain.
5. Keep inhibitory homeostasis and fixed paths outside this update.

Completion gate, using deterministic spike trains:

- positive dopamine potentiates causal aligned synapses;
- negative dopamine depresses the same aligned synapses;
- negative dopamine potentiates causal opposed synapses;
- positive dopamine depresses the same opposed synapses;
- neutral/fixed/homeostatic connections do not receive dopamine updates; and
- anti-causal eligibility reverses the corresponding results.

### Phase 5: build the reference TD architecture

Add a small architecture builder rather than duplicating population and
connection construction inside each experiment.

1. Construct positive/negative reward, positive/negative value, comparator
   relay, dopamine, hidden-principal, and hidden-inhibitory populations.
2. Give `DOPAMINE` tonic drive and a calibrated baseline observer.
3. Add an explicit transition-boundary API that samples current value and
   advances the previous-value trace exactly once.
4. Implement the six comparator terms from section 3.2 behind one interface.
5. Keep all reward/comparator coefficients fixed and non-scalable.
6. Feed dopamine deviation to the generalized plasticity rule.
7. Expose diagnostics for every individual TD term, both value rates, raw
   dopamine rate, baseline, and final modulation.

Completion gate:

- isolated synthetic rates reproduce the arithmetic TD target within a stated
  tolerance;
- repeated evaluation without advancing a transition cannot double-apply an
  outcome;
- reward, punishment, unexpectedly omitted reward, and accurate prediction
  produce the expected modulation signs;
- exact prediction produces near-baseline dopamine; and
- comparator output is independent of population size after calibration.

### Phase 6: test learning before neuralizing the trace

Use small purpose-built experiments before migrating the full conversation
tasks.

1. Verify recovery from silent and saturated initial conditions.
2. Verify positive and negative cue-value discrimination across multiple
   seeds.
3. Verify that a reward response transfers from outcome time toward the cue.
4. Verify that RPE at a predicted outcome shrinks rather than merely decays.
5. Trace individual `HIDDEN -> POSITIVE_VALUE` and
   `HIDDEN -> NEGATIVE_VALUE` strengths to detect random walk or winner-take-all
   behavior.
6. Verify action learning and value learning coexist in shared recurrent
   `HIDDEN` activity.

This phase contains a deliberate decision gate. External reward drives
`DOPAMINE`, but does not automatically create postsynaptic spikes in the value
assemblies. If value synapses still perform an unanchored random walk, stop and
test an explicit, correctly ordered teaching drive or dopamine-dependent
excitability for the matching value assembly. Do not hide the failure with
renormalization, larger populations, or a compatibility predictor.

Completion gate:

- content-specific value discrimination is statistically above chance;
- both appetitive and aversive associations learn with the expected pathway;
- learned strengths remain structured away from their bounds;
- zero-RPE trials cause no systematic drift; and
- results reproduce across a declared seed set.

### Phase 7: replace the previous-value trace with neurons

1. Preserve the comparator interface and reference implementation.
2. Implement transition-gated neural storage of positive and negative previous
   value. Use explicit excitatory/inhibitory relay populations for net signs.
3. Run reference and neural comparators on identical recorded rate streams.
4. Compare per-transition term values, dopamine traces, and learning outcomes.
5. Make the neural comparator the production implementation once it passes.

Do not call a fixed tick-delay chain complete for variable-duration tasks. The
stored value must advance on the same explicit transition event as the
reference trace.

Completion gate:

- the neural circuit matches reference RPE sign on every test transition;
- magnitude and timing remain within declared tolerances;
- no sustained oscillation or baseline drift appears after a phasic event;
- behavior is stable across task delays and seeds; and
- the reference comparator remains only as an oracle/test fixture, not a
  runtime fallback.

### Phase 8: migrate experiments and remove the legacy architecture

Primary files: `src/experiments/turn_taking.py`,
`src/experiments/discrimination.py`, diagnostics, package exports, and related
architecture documentation.

1. Migrate experiments to the shared architecture builder and explicit
   transition boundaries.
2. Replace signed reward-channel writes with magnitude routed to exactly one
   unipolar reward input.
3. Replace predictor diagnostics with value-path and dopamine diagnostics.
4. Remove experiment-specific renormalization of predictor/reward channels.
5. Delete `NumericChannel`, `NumericPopulationSpec`, the legacy arithmetic
   `RPE`, predictor population, signed-weight assumptions, stale parameters,
   and obsolete documentation. Do not leave aliases.
6. Update the general learning/reward architecture documents to match the
   implemented design.

Completion gate:

- no `REWARD + PREDICTOR` runtime path remains;
- no experiment computes RPE arithmetically outside the comparator oracle;
- no compatibility option can select the legacy architecture;
- static checks and all tests pass; and
- migrated experiments satisfy the Phase 6 learning criteria.

## 6. Validation matrix

Every run must record population rates, target rates, explicit drive currents,
strength distributions by connection, individual TD terms, dopamine rate, and
decoded modulation.

| Scenario | Required result |
| --- | --- |
| No task input | Stable sparse `HIDDEN`; tonic `DOPAMINE`; zero modulation |
| Bad low-activity initialization | Explicit homeostatic drive restores activity |
| Bad high-activity initialization | Inhibition and scaling prevent saturation |
| Unexpected positive reward | Dopamine burst; aligned causal learning |
| Unexpected negative reward | Dopamine dip; opposed causal learning |
| Predicted positive reward | Outcome RPE approaches zero |
| Predicted negative reward | Outcome RPE approaches zero |
| Omitted expected positive reward | Dopamine dip |
| Omitted expected negative reward | Dopamine burst |
| Long/variable state duration | Previous value advances only at boundary |
| Zero RPE over long run | No systematic strength or dopamine-baseline drift |
| Population-size sweep | Comparable decoded values and stable activity |
| Seed sweep | Same qualitative learning, no frequent dead/saturated seeds |

Passing a single seed or showing a visually plausible dopamine trace is not
sufficient. Report aggregate results and retain failing seeds for diagnosis.

## 7. Work-package rules for implementation agents

Each agent should receive one phase at a time plus this document. A phase is
not complete until its deletion work and completion gate are satisfied.

Agents must:

- prefer replacement over wrapping deprecated abstractions;
- remove dead callers, exports, parameters, comments, and diagnostics in the
  same phase that makes them obsolete;
- avoid adding feature flags for the old architecture;
- keep circuit construction in shared builders rather than copying it between
  experiments;
- add deterministic unit tests before empirical tuning;
- report architectural deviations instead of silently choosing the smallest
  patch; and
- leave population sizes, rates, gains, and timescales configurable, while
  keeping pathway signs and plasticity ownership structural.

An implementation that leaves `NumericChannel`, the arithmetic runtime `RPE`,
or indiscriminate network-wide STDP in place has not completed this redesign,
even if the new classes also exist.

## 8. Explicitly unresolved empirical question

The main remaining risk is eligibility bootstrapping in the two learned value
assemblies. Dopamine-opposed learning fixes the sign of negative-value updates,
but neither response profile guarantees that the correct value neurons fire in
the first place.

Phase 6 must establish whether shared `HIDDEN` drive plus sparse regulated
baseline activity provides enough causal structure. If it does not, the next
design should add an explicit local source of reward-correlated postsynaptic
activity and test it directly. This question must not be declared solved merely
because `DOPAMINE` itself bursts or dips.
