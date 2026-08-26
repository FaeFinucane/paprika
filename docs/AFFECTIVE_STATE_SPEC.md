# Affective State Reference

The bounded state variables and event updates described here are implemented in
`src/continual_agent/cognition/affect.py` and
`src/continual_agent/cognition/affect_circuit.py`. Design goals and deferred
effects are not runtime guarantees.

## Purpose

These variables are functional control state, not claims about subjective
experience. They are observable, bounded, and connected to behavior or
plasticity where noted. Human-readable labels are derived debug labels, not
primitive neurons.

Unless noted otherwise, values use `[0, 1]` and follow:

```text
state_next = clamp(state + rate * (target - state) + event_input, 0, 1)
```

`valence` uses `[-1, 1]`; `AffectiveState.advance()` decays transient values
toward baselines and consumes a small amount of energy.

## Signals

| Signal | Current implementation | Deferred effects |
| --- | --- | --- |
| Valence | Updated from reward-prediction error; represented in the affect circuit | No claim of subjective happiness |
| Arousal | Updated from urgency/threat and represented in the circuit | Threshold, attention, and deliberation changes |
| Uncertainty | Updated from uncertainty/correction and represented in the circuit | Automatic clarification or information seeking |
| Curiosity | Updated from novelty and learning progress; contributes to modulation | Safe exploration and attention allocation |
| Threat | Updated and represented in the circuit | Safety veto, refusal, and exploration suppression |
| Competence | Updated from prediction error and represented in the circuit | General confidence or persistence policy |
| Social affiliation | Updated from social feedback and represented in the circuit | Acknowledgement or repair policy |

`AffectiveState.modulation()` provides a global third-factor multiplier for
reward-modulated STDP. `AffectiveCircuit.align()` performs local feature/target
alignment on input-to-affect synapses, and `decode()` converts affect spikes to
inspectable values. The circuit does not run a second simulator.

## Deferred state and labels

`energy` currently decays but does not gate computation or enforce a budget.
Derived labels such as `happy_like`, `alarmed_like`, `exploratory_like`, and
`uncertain_like` are debug/evaluator composites and receive no independent
plasticity.

There is currently no normative veto or morality state. Future safety work must
use explicit permissions, safety, privacy, consent, and transparent objectives;
it must not be represented by an undefined universal goodness score.

See [architecture](ARCHITECTURE.md) for network placement and
[tasks](TASKS.md) for deferred work.
