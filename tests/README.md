# Test taxonomy

- `unit/`: deterministic local behavior and structural invariants. These do
  not depend on emergent network dynamics or seed sweeps.
- `viability/`: qualitative, multi-seed acceptance envelopes. The minimal E/I
  stability contract uses fixed E/I feedback, explicit homeostatic drives,
  and synaptic scaling, then perturbs transient
  membrane state directly. The dopamine scenario establishes a quiet tonic
  baseline, then tests reward/current-value signs, opposing previous-value
  signs, recovery, and numerical health across wiring seeds. Add the `slow`
  marker when a test is unsuitable for the default fast suite.
- `support/`: test-only scenario operations; production diagnostics and network
  setup remain under `src/`.

Use `pytest -m unit` or `pytest -m viability` to run one layer. Exploratory
parameter sweeps belong under `src/experiments`, not under `tests/`.
