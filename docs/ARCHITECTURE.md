# Synthetic temporal architecture

This repository focuses on a small, inspectable temporal-copy experiment. It
is not a conversational or language agent.

`PopulationLayout` defines population dimensions and named subgroup slices.
`NetworkConfig` carries the finalized layout and simulation settings.
`WeightInitializer` owns reproducible connectivity and bounded initialization.
`SpikingRuntime` is the composition root for the vectorised tick loop, drives,
readout, plasticity, diagnostics, ablation mask, and raw input interfaces.

`RuntimeSession` owns lifecycle, snapshots, isolation, and sparse weight-delta
merging. `InputRunner` owns raw input episodes and supervised or
reward-modulated training. `EventReadout` converts output spikes to timestamped
events, and `evaluate_event_stream` scores those events.

`synthetic_temporal.py` creates numeric frames with explicit input boundaries.
Each trial creates a fresh runtime, optionally ablates a pathway, trains it,
runs the stream, and evaluates actual output events. Immediate trials observe
character output during input; delayed trials begin observation at input end and
preserve the runtime clock. The supervised path uses structural teacher
alignment; the reward path reinforces observed event reports.

There is no text encoder, conversation facade, host-side prediction queue,
language feedback loop, or affective state in this workload.

The tests cover raw-frame training, session boundaries, readout arbitration,
event accounting, STDP, pathway controls, reproducibility, and experiment
summaries.
