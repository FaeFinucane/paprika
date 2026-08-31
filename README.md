# Continual Spiking Agent

This is a small, inspectable synthetic temporal-copy experiment built from
leaky integrate-and-fire neurons, sparse recurrent synapses, reward-modulated
STDP, named event readouts, and explicit session boundaries.

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

The experiment is deliberately non-linguistic: numeric frames are copied
through the spiking output populations and scored as timestamped events.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — network structure, populations, and
  implementation locations.
- [Network construction architecture](docs/remake/NETWORK_ARCHITECTURE.md) — population
  layouts, connectivity specifications, seeding, and compiled network outputs.
- [Training](docs/TRAINING.md) — raw-frame training and experiment controls.
- [Tasks](docs/TASKS.md) — deferred and actionable work.
- [Synthetic temporal experiments](docs/experiments/SYNTHETIC_TEMPORAL.md) — small
  immediate/delayed copy and pathway-control harness.

The commands above are the standard local verification set. The synthetic
experiment is the canonical architecture viability entry point.
