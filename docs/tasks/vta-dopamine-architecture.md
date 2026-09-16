# VTA dopamine architecture

The dopamine experiment uses a continuous cue-to-outcome spiking circuit. It
does not construct a scalar value function or calculate an arithmetic TD error
at runtime.

```text
CUE
  -> INFERRED_STATE_E <-> INFERRED_STATE_I
       -> VTA_DA
       -> TEMPORAL_0 -> ... -> TEMPORAL_N
                        -> VTA_INHIB -| VTA_DA

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

`add_attractor()` and `add_temporal_sequence()` in
`src/architectures/components.py` add reusable assemblies to a builder and
return name-based handles. This keeps every generated population and projection
visible to diagnostics and ablation experiments.

## Continuous operation

Trial boundaries never clear membrane voltage, queued current, recurrent
activity, or plasticity traces. Experiments stop their external drives and run
ordinary settling ticks. Residual state is observable network state, not an
episode-reset concern.

## Required checks

- cues recruit a bounded inferred-state assembly;
- temporal stages can recruit `VTA_INHIB`;
- `VTA_INHIB` suppresses `VTA_DA` through its learnable inhibitory projection;
- positive and negative uncued outcomes yield opposite DA deviations; and
- all old value, TD-comparator, and signed external TD-pulse code is absent.
