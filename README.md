# Continual Spiking Agent

The first vertical slice is a small, inspectable text-action learner. It uses
leaky integrate-and-fire neurons, sparse recurrent synapses, reward-modulated
STDP, a local Hebbian policy readout, and adaptive internal drives.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Verify

```bash
.venv/bin/pytest -q
```

## Run the automated curriculum

```bash
.venv/bin/python -m continual_agent.experiments.run_conversation
```

The current curriculum teaches five typed response intents:

```text
clarify, uncertain, answer, revise, acknowledge
```

It is deliberately not a free-form chatbot yet. The experiment is intended to
make the online learning dynamics visible before language generation is added.
