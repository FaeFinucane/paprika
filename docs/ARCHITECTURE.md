# Architecture

This repository is a small NumPy-based spiking conversational-agent prototype.
It is an experimental SNN, not a general language model. In particular, the
current event-training path is an initial recurrent-state baseline, not robust
general sequence memory or full sequence learning.

## Network

The runtime is one global recurrent graph of vectorised leaky integrate-and-fire
(LIF) neurons with sparse synapses. Synaptic current is applied one tick after a
spike. `PopulationLayout` is the immutable, validated source of population
bounds and semantic subgroups.

The default configuration has 188 neurons:

| Population | Size | Purpose | Implementation |
| --- | ---: | --- | --- |
| `INPUT` | 48 | Rate-coded text/action features | `src/continual_agent/encoding/text_encoder.py`; assembled in `src/continual_agent/agent/spiking_runtime.py` |
| `HIDDEN` | 48 | Recurrent internal state | `src/continual_agent/simulation/network.py`; layout in `src/continual_agent/simulation/population_layout.py` |
| `AFFECT` | 7 x 4 = 28 | Neural populations for valence, arousal, uncertainty, curiosity, threat, competence, and social affiliation | `src/continual_agent/cognition/affect_circuit.py` and `src/continual_agent/cognition/affect.py` |
| `OUTPUT_ACTION` | 7 x 4 = 28 | Typed action candidates | `src/continual_agent/cognition/readout.py`; constructed in `src/continual_agent/agent/spiking_runtime.py` |
| `OUTPUT_CHAR` | 12 x 3 = 36 | EOS plus the configured character alphabet | `src/continual_agent/language/spiking_decoder.py` |

`AgentConfig` controls these dimensions. `SpikingRuntime` constructs the layout,
LIF state, sparse projections, readout, and plasticity object. The canonical
`ConversationAgent` facade adds text encoding, affect, working memory, and task
training without duplicating runtime-owned fields. The layout is consumed by
projection helpers, readouts, plasticity targeting, and tests rather than
duplicated offsets.

## Projections and flow

- Text is converted to bounded rate-coded frames. Input dimensions 0 and 1 are
  reserved for neural `INPUT_BEGIN` and `INPUT_END` currents; text hashing uses
  only the remaining dimensions. Each boundary advances the network, rather
  than being only a `ResponseSession` check. `INPUT_END` is the first observable
  response tick in delayed mode, while semantic input output remains withheld.
- Input projects directly to action and affect populations. Affective
  populations project to action populations through ordinary recurrent synapses;
  there is no host-side affect-to-action arithmetic.
- Input, hidden, affect, and action activity can project to character output
  populations through the sparse graph. The hidden region is recurrent space,
  not a separate language module.
- `EventReadout` observes action or character evidence with separate activation
  and release thresholds. One output is globally active at a time; no other
  output is registered until that active population falls below the release
  threshold. Simultaneous candidates are deterministically arbitrated, and
  timestamps and inter-event intervals are recorded. Silence is not an event.
  `<EOS>` is a terminal character event.
- `ResponseSession` tracks input and response lifecycle. Readout state is
  response-local by default; recurrent neuron state, synaptic activity, affect,
  working memory, and eligibility persist according to `SessionPolicy`.

The lifecycle is:

```text
IDLE -> RECEIVING_INPUT -> RESPONDING -> EOS_RECEIVED -> COMPLETE
                                      \-> EXHAUSTED
```

`EXHAUSTED` means the configured output limit was reached without EOS. The
network is not reset between character events in a response.

## State and support components

- `src/continual_agent/simulation/neurons.py` implements LIF voltage,
  refractory state, and updates.
- `src/continual_agent/simulation/synapses.py` implements sparse edges and
  trainable weights.
- `src/continual_agent/cognition/working_memory.py` provides the optional
  host-side feature context controlled by `persistent_working_memory`; it is
  not a learned language-state population.
- `src/continual_agent/agent/session.py` implements lifecycle, snapshots,
  sparse weight deltas, and explicit merge policies. Runtime isolation is
  centralized in `SpikingRuntime`; task adapters transfer only their own state.
- `src/continual_agent/environment/protocol.py` and `scenarios.py` define typed
  actions and curriculum scenarios.
- `src/continual_agent/agent/debug.py` exposes inspectable response snapshots.
- `runtime_metrics.py` accumulates windowed population diagnostics. Active means
  at least one spike in the window, silent means zero spikes, and saturated means
  a per-neuron rate at or above the configured saturation rate (0.5 by default).
  `population_homeostasis.py` optionally applies a slow, bounded shared current
  from population mean-rate error; it never drives neurons toward identical rates.
- `src/continual_agent/agent/config.py`, `response.py`, and `event_training.py`
  keep configuration, response lifecycle, and event training separate from the
  public facade.

## Limitations

There is no dedicated learned context population, output feedback channel, or
output gate. Action selection comes from `OUTPUT_ACTION` through
`EventReadout`, not from a parallel host policy. Advanced plasticity schedules,
structural rewiring, individual adaptive thresholds, and network scaling are
deferred. Homeostasis is opt-in and population-level only, so its defaults do
not alter existing behavior.

The character path supports teacher-presented ordered events through existing
recurrence, including EOS, but this should not be described as full sequence
learning. Stronger dedicated tests for exact learned strings and repeated
characters remain needed.

See [training](TRAINING.md), [tasks](TASKS.md), and the
[affective-state reference](AFFECTIVE_STATE_SPEC.md).
