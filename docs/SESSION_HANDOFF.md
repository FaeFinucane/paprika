# Session Handoff

Status reflects committed code and the canonical experiment run on 2026-08-27.
This is a small NumPy SNN prototype, not a general sequence learner or chatbot.

## Architecture

- `NetworkCore` (`src/continual_agent/simulation/core.py`) owns vectorized LIF
  state, sparse synapses, one-tick delayed pending current, reset, and snapshots.
  It has reusable spike buffers, not spike history.
- `NetworkConfig` (`src/continual_agent/agent/network_config.py`) is the sole
  builder for `PopulationLayout`, projections, readout, plasticity, drives, and
  plugins, returned as one `NetworkBundle`.
- `SpikingRuntime` (`src/continual_agent/agent/spiking_runtime.py`) is the
  composition root and owns the tick loop, boundary currents, diagnostics, and
  ablation APIs. `RuntimeSession`
  (`src/continual_agent/agent/runtime_session.py`) owns lifecycle, snapshots,
  isolation, and sparse weight-delta merging. `InputRunner`
  (`src/continual_agent/agent/input_runner.py`) owns raw input episodes and
  supervised/reward training.
- `Drive`/`DriveAggregator` and `BackgroundDrive` live in
  `src/continual_agent/agent/drives.py`. `NetworkPlugin` plus metrics and
  plasticity hooks live in `src/continual_agent/agent/plugins.py`;
  `RuntimeMetrics` and optional population-level homeostasis are in
  `src/continual_agent/agent/runtime_metrics.py` and
  `src/continual_agent/agent/population_homeostasis.py`.
- Input dimensions 0 and 1 are neural `INPUT_BEGIN` and `INPUT_END` channels;
  text features use the remaining dimensions. There is no output feedback,
  output gate, timer mechanism, spike-history API, or legacy runtime alias.

## Results

Canonical command: `python -m continual_agent.experiments.synthetic_temporal`.
The current default run reports:

| Condition | Supervised | Reward-modulated STDP |
| --- | --- | --- |
| Immediate | recall/EOS `1.00/1.00`, `0.014` events/tick | `0.00/0.00`, `0.000` events/tick |
| Delayed | recall/EOS `1.00/1.00`, `0.014` events/tick | `0.00/0.00`, `0.000` events/tick |

Untrained, no-learning, shuffled-target, and recurrent-ablation controls were
`0.00/0.00`; direct input-to-output ablation was `0.38/0.38` (mean recall/EOS).
The supervised path is a structural teacher-alignment baseline. STDP currently
fails to produce a useful event stream, including immediate copy, so delayed
retention is not yet interpretable as an STDP result.

## Next Investigations

1. Run verbose trials and inspect reward, firing/event rates, eligibility, and
   direct/hidden/recurrent pathway weight changes before changing architecture.
2. Verify spike timing and eligibility coverage through `INPUT_BEGIN`, frames,
   `INPUT_END`, and response ticks; check whether the zero-event path leaves
   eligible edges and whether reward is committed after the observed outcome.
3. Sweep the existing early/reliable reward schedules, background drive, output
   thresholds, and response duration with fixed seeds. Compare with supervised
   and no-learning controls.
4. Add focused tests for `ma`, `ba`, `mama`, `baba`, repeated characters, silence,
   premature/missing EOS, and post-EOS suppression after the failure is defined.

## Tasks And Deferrals

Active work is event-stream measurement/capacity, deciding whether a learned
context population is justified, and stronger event-stream tests. Keep the
typed-action curriculum explicitly named as a baseline. Do not scale the network
until measurements show a bottleneck.

Deferred features are output reafferent feedback, post-selection output gating,
intrinsic oscillatory/pacemaker research, structural rewiring, adaptive
thresholds, and normative/resource-budget policies. Population homeostasis is
opt-in, slow, bounded, shared per population, and disabled by default.

## Verification

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m continual_agent.experiments.synthetic_temporal
```

See [Architecture](ARCHITECTURE.md), [Training](TRAINING.md), [Tasks](TASKS.md),
and [Synthetic Temporal](experiments/SYNTHETIC_TEMPORAL.md).
