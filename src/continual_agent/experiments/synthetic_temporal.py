"""Fresh-agent paired experiments for structural and reward-modulated learning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from continual_agent.agent.session import InputSignal
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.cognition.readout import OutputEvent
from continual_agent.evaluation.event_stream import (
    EventStreamReport,
    RewardSchedule,
    evaluate_event_stream,
)
from continual_agent.simulation.population_layout import Population


class CopyCondition(str, Enum):
    IMMEDIATE = "immediate-copy"
    DELAYED = "delayed-copy"


class AgentKind(str, Enum):
    SUPERVISED = "supervised-structural"
    REWARD_MODULATED_STDP = "reward-modulated-stdp"


class TrainingMode(str, Enum):
    UNTRAINED = "untrained"
    NO_LEARNING = "no-learning"
    SUPERVISED = "supervised"
    REWARD_MODULATED_STDP = "reward-modulated-stdp"


class Control(str, Enum):
    TRAINED = "trained"
    UNTRAINED = "untrained"
    NO_LEARNING = "no-learning"
    SHUFFLED_TARGET = "shuffled-target"
    RECURRENT_ABLATION = "recurrent-ablation"
    DIRECT_INPUT_OUTPUT_ABLATION = "direct-input-to-output-ablation"


@dataclass(frozen=True)
class TemporalExperimentConfig:
    alphabet: tuple[str, ...] = ("A", "B")
    sequences: tuple[tuple[str, ...], ...] = (("A",), ("B",))
    evaluation_sequences: tuple[tuple[str, ...], ...] | None = None
    training_trials: int = 12
    stdp_training_trials: int = 200
    ticks_per_frame: int = 3
    patience_window: int = 24
    response_ticks: int = 64
    hidden_neurons: int = 12
    seed: int = 0
    background_rate: float = 0.01
    background_current: float = 0.3
    stdp_background_rate: float = 0.05
    stdp_background_current: float = 0.5
    reward_stage: str = "early"
    homeostasis_enabled: bool = False
    homeostasis_target_rate: float = 0.1
    homeostasis_strength: float = 0.01
    homeostasis_update_interval: int = 100
    homeostasis_max_current: float = 0.25

    def __post_init__(self) -> None:
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet must contain distinct symbols")
        all_sequences = (*self.sequences, *(self.evaluation_sequences or ()))
        if any(symbol not in self.alphabet for sequence in all_sequences for symbol in sequence):
            raise ValueError("sequences must use the configured alphabet")
        if (
            self.training_trials < 0
            or self.stdp_training_trials < 0
            or self.ticks_per_frame <= 0
            or self.response_ticks <= 0
        ):
            raise ValueError("trial and timing values must be positive")
        if not 0.0 <= self.stdp_background_rate <= 1.0:
            raise ValueError("stdp_background_rate must be in [0, 1]")
        if self.stdp_background_current < 0.0:
            raise ValueError("stdp_background_current must be non-negative")
        RewardSchedule.for_stage(self.reward_stage)


@dataclass(frozen=True)
class TemporalTrialResult:
    condition: CopyCondition
    agent: AgentKind
    control: Control
    training_mode: TrainingMode
    target: tuple[str, ...]
    observed: tuple[OutputEvent, ...]
    report: EventStreamReport
    hidden_activity: float
    weight_change: float
    reward: float = 0.0
    eligibility_change: float = 0.0
    pathway_weight_change: dict[str, float] | None = None
    pathway_eligibility: dict[str, float] | None = None
    firing_rate: float = 0.0
    output_event_rate: float = 0.0
    population_diagnostics: dict[str, dict[str, float]] | None = None


@dataclass
class TemporalExperimentResult:
    trials: list[TemporalTrialResult]

    @property
    def by_condition(self) -> dict[tuple[CopyCondition, Control], list[TemporalTrialResult]]:
        keys = {(trial.condition, trial.control) for trial in self.trials}
        return {
            key: [trial for trial in self.trials if (trial.condition, trial.control) == key]
            for key in sorted(
                keys,
                key=lambda item: (
                    list(CopyCondition).index(item[0]),
                    list(Control).index(item[1]),
                ),
            )
        }

    def summary(self) -> dict[tuple[str, str, str], dict[str, float]]:
        result: dict[tuple[str, str, str], dict[str, float]] = {}
        keys = {(x.condition, x.agent, x.control) for x in self.trials}
        ordered_keys = sorted(
            keys,
            key=lambda item: (
                list(CopyCondition).index(item[0]),
                list(AgentKind).index(item[1]),
                list(Control).index(item[2]),
            ),
        )
        for condition, agent, control in ordered_keys:
            trials = [
                x
                for x in self.trials
                if (x.condition, x.agent, x.control) == (condition, agent, control)
            ]
            metrics = {
                "event_precision": float(np.mean([x.report.event_precision for x in trials])),
                "event_recall": float(np.mean([x.report.event_recall for x in trials])),
                "eos_accuracy": float(np.mean([x.report.eos_accuracy for x in trials])),
                "mean_latency": float(np.mean([x for t in trials for x in t.report.latencies]))
                if any(t.report.latencies for t in trials)
                else 0.0,
                "hidden_activity": float(np.mean([x.hidden_activity for x in trials])),
                "weight_change": float(np.mean([x.weight_change for x in trials])),
                "reward": float(np.mean([x.reward for x in trials])),
                "eligibility_change": float(np.mean([x.eligibility_change for x in trials])),
                "firing_rate": float(np.mean([x.firing_rate for x in trials])),
                "output_event_rate": float(np.mean([x.output_event_rate for x in trials])),
            }
            for pathway in ("direct_input_output", "hidden_output", "recurrent_event"):
                metrics[f"weight_{pathway}"] = float(
                    np.mean([(x.pathway_weight_change or {}).get(pathway, 0.0) for x in trials])
                )
                metrics[f"eligibility_{pathway}"] = float(
                    np.mean([(x.pathway_eligibility or {}).get(pathway, 0.0) for x in trials])
                )
            result[(condition.value, agent.value, control.value)] = metrics
        return result


def _frame(config: TemporalExperimentConfig, index: int, value: float = 5.0) -> np.ndarray:
    frame = np.zeros(len(config.alphabet) + 2)
    frame[index] = value
    return frame


def _stream(
    config: TemporalExperimentConfig, sequence: tuple[str, ...]
) -> tuple[list[InputSignal | np.ndarray], list[tuple[np.ndarray, str]]]:
    events: list[InputSignal | np.ndarray] = [InputSignal.INPUT_BEGIN]
    labelled: list[tuple[np.ndarray, str]] = []
    for symbol in sequence:
        frame = _frame(config, 2 + config.alphabet.index(symbol))
        events.append(frame)
        labelled.append((frame, symbol))
        events.extend(np.zeros_like(frame) for _ in range(config.ticks_per_frame))
    events.append(InputSignal.INPUT_END)
    return events, labelled


def _supervised_labels(
    config: TemporalExperimentConfig, sequence: tuple[str, ...]
) -> list[tuple[np.ndarray, str]]:
    labelled = list(_stream(config, sequence)[1])
    return labelled


def _make_runtime(
    config: TemporalExperimentConfig, seed: int, *, stdp_background: bool = False
) -> SpikingRuntime:
    background_rate = config.stdp_background_rate if stdp_background else config.background_rate
    background_current = (
        config.stdp_background_current if stdp_background else config.background_current
    )
    agent = SpikingRuntime(
        input_features=len(config.alphabet) + 2,
        hidden_neurons=config.hidden_neurons,
        output_tokens=("<EOS>", *config.alphabet),
        neurons_per_token=2,
        seed=seed,
        background_rate=background_rate,
        background_current=background_current,
        homeostasis_enabled=config.homeostasis_enabled,
        homeostasis_target_rate=config.homeostasis_target_rate,
        homeostasis_strength=config.homeostasis_strength,
        homeostasis_update_interval=config.homeostasis_update_interval,
        homeostasis_max_current=config.homeostasis_max_current,
    )
    agent.output_readout.activation_threshold = 0.25
    agent.output_readout.release_threshold = 0.2
    agent.network.neurons.tau_membrane = 3.0
    agent.network.neurons.refractory_ticks = 1
    return agent


def _train(
    agent: SpikingRuntime,
    input_events: list[InputSignal | np.ndarray],
    labelled: list[tuple[np.ndarray, str]],
    kind: AgentKind,
    mode: TrainingMode,
    trials: int,
    condition: CopyCondition,
    config: TemporalExperimentConfig,
    target: tuple[str, ...],
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    if mode in (TrainingMode.UNTRAINED, TrainingMode.NO_LEARNING):
        return 0.0, 0.0, {}, {}
    reward = 0.0
    eligibility_change = 0.0
    pathway_change = {"direct_input_output": 0.0, "hidden_output": 0.0, "recurrent_event": 0.0}
    pathway_eligibility = {name: 0.0 for name in pathway_change}
    reward_config = RewardSchedule.for_stage(config.reward_stage).event_config(
        patience_window=config.patience_window
    )
    for _ in range(trials):
        events: list[InputSignal | tuple[np.ndarray, str]] = [
            InputSignal.INPUT_BEGIN,
            *labelled,
            InputSignal.INPUT_END,
        ]
        if kind is AgentKind.SUPERVISED:
            agent.train_input_events(events)
        else:
            before_weights = agent.network.synapses.weight.copy()

            immediate_reward = 0.0
            position = 0

            def reward_event(event: OutputEvent) -> None:
                nonlocal immediate_reward, position
                expected = target[position] if position < len(target) else None
                if event.name == expected:
                    value = reward_config.correct_reward
                    position += 1
                elif event.name == "<EOS>" and expected != "<EOS>":
                    value = reward_config.premature_eos_reward
                else:
                    value = reward_config.incorrect_reward
                immediate_reward += value
                agent.plasticity.reinforce(value)

            observed = agent.run_input_events(
                input_events,
                response_ticks=config.response_ticks,
                observe_during_input=condition is CopyCondition.IMMEDIATE,
                reward_callback=reward_event,
            )
            report = evaluate_event_stream(
                target,
                observed,
                config=reward_config,
                end_time=sum(isinstance(item, np.ndarray) for item in input_events)
                + config.response_ticks,
            )
            reward += report.total_reward
            eligibility_change += float(np.abs(agent.plasticity.eligibility).sum())
            for name, indices in (
                ("direct_input_output", agent.direct_input_output_edge_indices),
                ("hidden_output", agent.hidden_output_edge_indices),
                ("recurrent_event", agent.recurrent_event_edge_indices),
            ):
                pathway_eligibility[name] += float(
                    np.abs(agent.plasticity.eligibility[indices]).sum()
                )
            # Event rewards are committed as soon as the network emits. Any
            # remaining score covers missing/deferred outcomes.
            agent.plasticity.reinforce(report.total_reward - immediate_reward)
            changed = agent.network.synapses.weight - before_weights
            for name, indices in (
                ("direct_input_output", agent.direct_input_output_edge_indices),
                ("hidden_output", agent.hidden_output_edge_indices),
                ("recurrent_event", agent.recurrent_event_edge_indices),
            ):
                pathway_change[name] += float(np.abs(changed[indices]).sum())
            agent.plasticity.reset_traces()
    return reward, eligibility_change, pathway_change, pathway_eligibility


def run_temporal_experiment(
    config: TemporalExperimentConfig | None = None,
    *,
    conditions: Iterable[CopyCondition] = tuple(CopyCondition),
    controls: Iterable[Control] = (
        Control.TRAINED,
        Control.UNTRAINED,
        Control.NO_LEARNING,
        Control.SHUFFLED_TARGET,
        Control.RECURRENT_ABLATION,
        Control.DIRECT_INPUT_OUTPUT_ABLATION,
    ),
    agents: Iterable[AgentKind] = (AgentKind.SUPERVISED,),
) -> TemporalExperimentResult:
    """Run fresh, independent runtime cells; no sequence is retained by the host."""
    config = config or TemporalExperimentConfig()
    evaluation_sequences = config.evaluation_sequences or config.sequences
    trials: list[TemporalTrialResult] = []
    for condition in conditions:
        for kind in agents:
            for control in controls:
                for trial_index, sequence in enumerate(evaluation_sequences):
                    seed = (
                        config.seed
                        + trial_index
                        + 1000 * list(CopyCondition).index(condition)
                        + 10000 * list(Control).index(control)
                        + 100000 * list(AgentKind).index(kind)
                    )
                    agent = _make_runtime(
                        config,
                        seed,
                        stdp_background=kind is AgentKind.REWARD_MODULATED_STDP,
                    )
                    if control is Control.RECURRENT_ABLATION:
                        agent.ablate_edges(agent.hidden_output_edge_indices)
                    if control is Control.DIRECT_INPUT_OUTPUT_ABLATION:
                        agent.ablate_edges(agent.direct_input_output_edge_indices)
                    events, labelled = _stream(config, sequence)
                    target = (*sequence, "<EOS>")
                    before = agent.network.synapses.weight.copy()
                    mode = (
                        TrainingMode.UNTRAINED
                        if control is Control.UNTRAINED
                        else TrainingMode.NO_LEARNING
                        if control is Control.NO_LEARNING
                        else TrainingMode.REWARD_MODULATED_STDP
                        if kind is AgentKind.REWARD_MODULATED_STDP
                        else TrainingMode.SUPERVISED
                    )
                    if control not in (Control.UNTRAINED, Control.NO_LEARNING):
                        training_target = (
                            _supervised_labels(config, sequence)
                            if control is not Control.SHUFFLED_TARGET
                            else [
                                (frame, label)
                                for (frame, _), label in zip(
                                    _supervised_labels(config, sequence), reversed(target)
                                )
                            ]
                        )
                        reward, eligibility_change, pathway_change, pathway_eligibility = _train(
                            agent,
                            events,
                            training_target,
                            kind,
                            mode,
                            (
                                config.stdp_training_trials
                                if kind is AgentKind.REWARD_MODULATED_STDP
                                else config.training_trials
                            ),
                            condition,
                            config,
                            tuple(label for _, label in training_target),
                        )
                    else:
                        reward, eligibility_change, pathway_change, pathway_eligibility = (
                            0.0,
                            0.0,
                            {},
                            {},
                        )
                    agent.reset_diagnostics()
                    observed = agent.run_input_events(
                        events,
                        response_ticks=config.response_ticks,
                        observe_during_input=condition is CopyCondition.IMMEDIATE,
                    )
                    hidden = agent.layout.slice(Population.HIDDEN)
                    report = evaluate_event_stream(
                        target,
                        observed,
                        config=RewardSchedule.for_stage(config.reward_stage).event_config(
                            patience_window=config.patience_window
                        ),
                        end_time=sum(isinstance(item, np.ndarray) for item in events)
                        + config.response_ticks,
                        withheld_prefix=0,
                    )
                    trials.append(
                        TemporalTrialResult(
                            condition,
                            kind,
                            control,
                            mode,
                            target,
                            observed,
                            report,
                            float(np.mean(np.abs(agent.network.neurons.voltage[hidden]))),
                            float(np.abs(agent.network.synapses.weight - before).sum()),
                            reward,
                            eligibility_change,
                            pathway_change,
                            pathway_eligibility,
                            agent.diagnostics["firing_rate"],
                            agent.diagnostics["output_event_rate"],
                            agent.population_diagnostics,
                        )
                    )
    return TemporalExperimentResult(trials)


def run_paired_temporal_experiment(
    config: TemporalExperimentConfig | None = None,
) -> TemporalExperimentResult:
    """Run the supervised and reward-modulated agents over identical fresh trials."""
    return run_temporal_experiment(config, agents=tuple(AgentKind))


def _label(value: Enum) -> str:
    if value is AgentKind.SUPERVISED:
        return "Supervised"
    if value is AgentKind.REWARD_MODULATED_STDP:
        return "STDP"
    return value.value.replace("-copy", "").replace("-", " ").title()


def _format_row(label: str, values: dict[str, float]) -> str:
    return (
        f"{label:<12} recall={values['event_recall']:.2f} eos={values['eos_accuracy']:.2f} "
        f"rate={values.get('firing_rate', 0.0):.3f}/"
        f"{values.get('output_event_rate', 0.0):.3f}"
    )


def _print_summary(result: TemporalExperimentResult, *, verbose: bool) -> None:
    summary = result.summary()
    if verbose:
        print("Synthetic temporal experiment (all aggregates)")
        for key, values in summary.items():
            condition, agent, control = key
            print(
                f"{_label(CopyCondition(condition))} / {_label(AgentKind(agent))} / "
                f"{_label(Control(control))}: recall={values['event_recall']:.2f} "
                f"eos={values['eos_accuracy']:.2f} "
                f"rate={values.get('firing_rate', 0.0):.3f}/"
                f"{values.get('output_event_rate', 0.0):.3f}"
            )
        return

    print("Synthetic temporal experiment")
    print("Trained comparison (recall / EOS)")
    for condition in CopyCondition:
        rows = [
            summary.get((condition.value, agent.value, Control.TRAINED.value))
            for agent in AgentKind
        ]
        if any(row is not None for row in rows):
            labels = []
            for agent, row in zip(AgentKind, rows):
                if row is not None:
                    labels.append(
                        f"{_label(agent)}={row['event_recall']:.2f}/{row['eos_accuracy']:.2f} "
                        f"{row.get('output_event_rate', 0.0):.3f}ev/t"
                    )
            print(f"{_label(condition):<12} " + "  ".join(labels))

    print("Control checks (mean across conditions and agents; recall / EOS)")
    for control in Control:
        if control is Control.TRAINED:
            continue
        control_rows = [
            values
            for (condition, agent, row_control), values in summary.items()
            if row_control == control.value
        ]
        if control_rows:
            control_values = {
                metric: float(np.mean([row[metric] for row in control_rows]))
                for metric in ("event_recall", "eos_accuracy")
            }
            print(_format_row(_label(control), control_values))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the paired synthetic temporal experiment")
    parser.add_argument(
        "--verbose", action="store_true", help="print every condition/agent/control aggregate"
    )
    args = parser.parse_args(argv)
    _print_summary(run_paired_temporal_experiment(), verbose=args.verbose)


if __name__ == "__main__":
    main()
