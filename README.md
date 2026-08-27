# Continual Spiking Agent

The first vertical slice is a small, inspectable text-action learner. It uses
leaky integrate-and-fire neurons, sparse recurrent synapses, reward-modulated
STDP, named event readouts, and bounded affective state.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Verify

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m continual_agent.experiments.synthetic_temporal
```

It is deliberately not a free-form chatbot yet. Character responses use the
spiking output populations and timestamped event-readout path.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — network structure, populations, and
  implementation locations.
- [Training](docs/TRAINING.md) — the separate training components and their
  implementations.
- [Tasks](docs/TASKS.md) — deferred and actionable work.
- [Affective state reference](docs/AFFECTIVE_STATE_SPEC.md) — bounded affective
  state and its current guarantees.
- [Synthetic temporal experiments](docs/experiments/SYNTHETIC_TEMPORAL.md) — small
  immediate/delayed copy and pathway-control harness.

The commands above are the standard local verification set. The synthetic
experiment is the canonical architecture viability entry point.
