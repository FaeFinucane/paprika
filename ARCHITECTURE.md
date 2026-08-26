# Network Architecture

This repository contains a small NumPy-based spiking conversational-agent
prototype. It is an experimental SNN, not yet a general language model.

## Current network

The simulator uses vectorised leaky integrate-and-fire neurons with sparse
synapses. Synaptic current is applied one tick after a spike. The network is a
single global recurrent graph; populations are currently represented mostly by
contiguous index ranges and calculated offsets.

Default layout:

```text
input          48
hidden         48
affect          7 x 4  = 28
actions         7 x 4  = 28
tokens          12 x 3 = 36  (EOS plus alphabet)
total                    188 neurons
```

The hidden region is implicit recurrent space rather than a dedicated module.
Explicit projections connect input to affect/actions, affect to actions, and
pre-token neurons to token populations. Token decoding groups several neurons
per symbol and uses spike counts over a decoding window.

## Main modules

- `src/continual_agent/simulation/neurons.py` — LIF neuron state and updates.
- `src/continual_agent/simulation/synapses.py` — sparse edges and weights.
- `src/continual_agent/simulation/network.py` — global recurrent simulation,
  spike history, and reset behaviour.
- `src/continual_agent/plasticity/stdp.py` — eligibility-trace,
  reward-modulated STDP.
- `src/continual_agent/language/spiking_decoder.py` — token populations,
  spike-count decoding, EOS handling, and generation.
- `src/continual_agent/cognition/readout.py` — action/affect readouts.
- `src/continual_agent/cognition/working_memory.py` — host-side decaying
  input vector; a prototype scaffold, not learned neural memory.
- `src/continual_agent/cognition/affect.py` and `affect_circuit.py` — bounded
  host affect state and neural affect populations.
- `src/continual_agent/agent/conversation_agent.py` — configuration,
  population construction, training, and conversation orchestration.

## Important current limitations

- Population offsets are passed around manually; a named population/layout
  abstraction is planned and should be introduced before further growth.
- There is no dedicated learned language/context-state population.
- Generation does not yet provide robust recurrent sequence memory; previous
  token/context handling is partly external scaffolding.
- Training does not clearly distinguish correct output, incorrect output,
  missed output, and unwanted output. Silence may therefore be rewarded too
  favourably depending on the training path.
- Host-side policy and first-token scoring overrides currently influence
  behaviour more strongly than some neural readouts. Treat these as scaffolding
  when evaluating what the SNN has learned.
- Plasticity has STDP and reward modulation, but no structural rewiring or
  active homeostasis/adaptive thresholds.

## Recommended direction

1. Add a central named population layout (`INPUT`, `HIDDEN`, `AFFECT`,
   `ACTION`, `LANGUAGE_STATE`, `TOKEN`) with safe slice/index selection.
2. Add a dedicated language-state population and explicit recurrent token
   transitions.
3. Decode one token per fixed simulation window, preserving recurrent state
   between windows while isolating spike counts. Use EOS and maximum length;
   do not rely primarily on silence as a delimiter.
4. Track positive reward for correct output and explicit penalties for wrong,
   missing, or post-EOS output.
5. Add configurable plasticity schedules and homeostasis before introducing
   structural synapse formation.
6. Compare recurrent neural memory against `WorkingMemory`, then reduce the
   host-side scaffold if the learned mechanism succeeds.
7. Increase network size only after exact sequence tests and activity/capacity
   measurements demonstrate that architecture, rather than training protocol,
   is the bottleneck.

## Useful verification

The current test suite contains 26 tests covering simulation, decoder bounds,
working memory, affect causality, and vertical integration. Exact learned
sequence behaviour (`ma`, `ba`, `mama`, `baba`, repeated characters, and EOS)
still needs stronger dedicated tests.

Run:

```bash
python -m pytest -q
```
