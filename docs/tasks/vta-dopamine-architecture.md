# VTA dopamine architecture

The dopamine experiment uses a continuous cue-to-outcome spiking circuit. It
does not construct a scalar value function or calculate an arithmetic TD error
at runtime.

```text
CUE
  -> INFERRED_STATE_E <-> INFERRED_STATE_I
       -> VTA_DA
       -> TEMPORAL_0_E -> ... -> TEMPORAL_7_E
          |                  |
       TEMPORAL_0_I       TEMPORAL_7_I
          \                  /
           ------> VTA_INHIB -| VTA_DA

OUTCOME_POSITIVE / OUTCOME_NEGATIVE -> VTA_DA
```

`VTA_DA` is modulatory-only. Its tonic-rate deviation is the third factor for
`DopamineSTDP`; outcome inputs influence learning only by changing that
dopamine signal. The outcome projections themselves are fixed. State, temporal,
and inhibitory pathways use named learning policies and bounded strengths.

## Construction

`NetworkBuilder` in `src/builder.py` owns mutable population, connection, and
population-plugin declarations. `compile(seed)` produces a ready `Session` and
instantiates declarative tonic drive, homeostatic drive, rate-observer, and
synaptic-scaling plugins with reproducible RNG state.

Circuits in `src/circuits/` own only their internal populations and intrinsic
connections, then expose named input/output handles. A parent assembly owns
every projection that crosses a circuit boundary. Thus the fixed conditioning
assembly in `src/experiments/dopamine.py` explicitly wires cue to inferred
state, state to the temporal basis and VTA-DA, temporal stages to `VTA_INHIB`,
and signed outcomes to VTA-DA. The reusable VTA component itself owns only
`VTA_INHIB`, `VTA_DA`, their intrinsic inhibitory projection, and the
session-level DA readout/plasticity installation.

The dopamine experiment has one explicit timing design: a cue-seeded E/I
temporal basis. Its state-entry and forward links are a stable causal scaffold;
local E/I regulation and each stage's readout to `VTA_INHIB` remain learnable.
This lets fixed-delay acquisition select cue-relative states without using
reward prediction errors to rewrite the time representation itself. The generic
asymmetric recurrent circuit remains independently available for experiments,
but is not an alternate dopamine-circuit mode.

## Continuous operation

Trial boundaries never clear membrane voltage, queued current, recurrent
activity, or plasticity traces. Experiments stop their external drives and run
ordinary settling ticks. Residual state is observable network state, not an
episode-reset concern.

## Required checks

- cues recruit a bounded inferred-state assembly;
- sparse state seeding recruits a bounded temporal-basis trajectory that fades
  after withdrawal;
- fixed-delay acquisition has an outcome-aligned `VTA_INHIB` peak supplied by
  `TEMPORAL_<stage>_E -> VTA_INHIB` readouts;
- `VTA_INHIB` suppresses `VTA_DA` through its learnable inhibitory projection;
- positive and negative uncued outcomes yield opposite DA deviations; and
- all old value, TD-comparator, and signed external TD-pulse code is absent.
